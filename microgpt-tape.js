'use strict';

// ══════════════════════════════════════════════
// MICROGPT TAPE — Autograd engine + transformer
// Ported from tapegpt.py
// https://gist.github.com/mplekh/3afbdfb9f063cf531cfd3d00685cfdc0
// ══════════════════════════════════════════════

// ── Tape class (pre-allocated typed arrays) ──

const TAPE_CAP = 80000;

class Tape {
  constructor() {
    this.data = new Float64Array(TAPE_CAP);
    this.grad = new Float64Array(TAPE_CAP);
    this.c1   = new Int32Array(TAPE_CAP);   // child 1 index (-1 = none)
    this.c2   = new Int32Array(TAPE_CAP);   // child 2 index (-1 = none)
    this.lg1  = new Float64Array(TAPE_CAP); // local grad w.r.t. c1
    this.lg2  = new Float64Array(TAPE_CAP); // local grad w.r.t. c2
    this.size = 0;
    this.c1.fill(-1);
    this.c2.fill(-1);
  }

  val(x) {
    const i = this.size++;
    this.data[i] = x;
    this.grad[i] = 0;
    this.c1[i] = -1;
    this.c2[i] = -1;
    this.lg1[i] = 0;
    this.lg2[i] = 0;
    return i;
  }

  push(val, c1, c2, lg1, lg2) {
    const i = this.size++;
    this.data[i] = val;
    this.grad[i] = 0;
    this.c1[i] = c1;
    this.c2[i] = c2;
    this.lg1[i] = lg1;
    this.lg2[i] = lg2;
    return i;
  }

  truncate(n) { this.size = n; }

  zero_grad() { this.grad.fill(0, 0, this.size); }

  backward(root) {
    this.grad[root] = 1.0;
    for (let i = root; i >= 0; i--) {
      const g = this.grad[i];
      if (g === 0) continue;
      const a = this.c1[i];
      if (a >= 0) {
        this.grad[a] += this.lg1[i] * g;
        const b = this.c2[i];
        if (b >= 0) this.grad[b] += this.lg2[i] * g;
      }
    }
  }
}

// ── Tape primitives ──

function t_add(t, a, b) {
  return t.push(t.data[a] + t.data[b], a, b, 1.0, 1.0);
}

function t_mul(t, a, b) {
  return t.push(t.data[a] * t.data[b], a, b, t.data[b], t.data[a]);
}

function t_mul_const(t, a, c) {
  return t.push(t.data[a] * c, a, -1, c, 0);
}

function t_pow_const(t, a, p) {
  const v = t.data[a];
  return t.push(v ** p, a, -1, v !== 0 ? p * v ** (p - 1) : 0, 0);
}

function t_div(t, a, b) {
  return t_mul(t, a, t_pow_const(t, b, -1));
}

function t_log(t, a) {
  const v = Math.max(1e-10, t.data[a]);
  return t.push(Math.log(v), a, -1, 1.0 / v, 0);
}

function t_relu(t, a) {
  const v = t.data[a];
  return t.push(v > 0 ? v : 0, a, -1, v > 0 ? 1 : 0, 0);
}

function t_sigmoid(t, a) {
  const v = t.data[a];
  const s = 1 / (1 + Math.exp(-v));
  return t.push(s, a, -1, s * (1 - s), 0);
}

function t_tanh(t, a) {
  const v = t.data[a];
  const th = Math.tanh(v);
  return t.push(th, a, -1, 1 - th * th, 0);
}

// ── Transformer components ──

function t_linear(t, x, wBase, outN, inN) {
  // x: Int32Array of tape indices (length inN)
  // wBase: starting tape index of flat row-major weight matrix (outN × inN)
  // Returns Int32Array of output indices (length outN)
  const out = new Int32Array(outN);
  for (let o = 0; o < outN; o++) {
    let acc = t.val(0);
    const rowBase = wBase + o * inN;
    for (let i = 0; i < inN; i++) {
      acc = t_add(t, acc, t_mul(t, x[i], rowBase + i));
    }
    out[o] = acc;
  }
  return out;
}

function t_rmsnorm(t, x, n) {
  let ss = t.val(0);
  for (let i = 0; i < n; i++) ss = t_add(t, ss, t_mul(t, x[i], x[i]));
  const ms = t_div(t, ss, t.val(n));
  const ms_e = t_add(t, ms, t.val(1e-5));
  const inv_std = t_pow_const(t, ms_e, -0.5);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = t_mul(t, x[i], inv_std);
  return out;
}

function t_softmax(t, logits, n) {
  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (t.data[logits[i]] > maxVal) maxVal = t.data[logits[i]];
  }
  const exps = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const shifted = t.push(t.data[logits[i]] - maxVal, logits[i], -1, 1.0, 0);
    const ev = Math.exp(t.data[shifted]);
    exps[i] = t.push(ev, shifted, -1, ev, 0);
  }
  let total = t.val(0);
  for (let i = 0; i < n; i++) total = t_add(t, total, exps[i]);
  const inv = t_pow_const(t, total, -1);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = t_mul(t, exps[i], inv);
  return out;
}

// ── GPT config ──

const GPT = {
  n_embd: 32, n_head: 4, head_dim: 8,
  n_layer: 1, block_size: 16,
  feat_dim: 208, n_actions: 4, mlp_mult: 4,
};

// ── GPT forward pass ──
// inputIndices: Int32Array(feat_dim) — tape indices for input features
// posId: integer position in sequence
// kvFloats: { keys: Float64Array, vals: Float64Array } — detached KV cache
// kvLen: number of cached past positions
// params: { w_in, wpe, wq, wk, wv, wo, f1, f2, w_policy, w_value } — base tape indices
// cfg: GPT config
// Returns: { policy: Int32Array(n_actions), value: tapeIndex, newK: Float64Array, newV: Float64Array }

function gpt_forward_full(t, inputIndices, posId, kvFloats, kvLen, params, cfg) {
  const { n_embd, n_head, head_dim, block_size } = cfg;

  // 1. Input projection
  let x = t_linear(t, inputIndices, params.w_in, n_embd, cfg.feat_dim);

  // 2. Positional embedding + RMSNorm
  const posBase = params.wpe + posId * n_embd;
  const xp = new Int32Array(n_embd);
  for (let i = 0; i < n_embd; i++) xp[i] = t_add(t, x[i], posBase + i);
  x = t_rmsnorm(t, xp, n_embd);

  // Store per-layer K/V indices for extraction
  let layerK, layerV;

  // 3. Transformer layers
  for (let li = 0; li < cfg.n_layer; li++) {
    const x_norm = t_rmsnorm(t, x, n_embd);
    const q = t_linear(t, x_norm, params.wq, n_embd, n_embd);
    const k = t_linear(t, x_norm, params.wk, n_embd, n_embd);
    const v = t_linear(t, x_norm, params.wv, n_embd, n_embd);
    layerK = k;
    layerV = v;

    // Past KV as detached leaves
    const allK = [], allV = [];
    for (let p = 0; p < kvLen; p++) {
      const kp = new Int32Array(n_embd);
      const vp = new Int32Array(n_embd);
      const off = li * block_size * n_embd + p * n_embd;
      for (let i = 0; i < n_embd; i++) {
        kp[i] = t.val(kvFloats.keys[off + i]);
        vp[i] = t.val(kvFloats.vals[off + i]);
      }
      allK.push(kp);
      allV.push(vp);
    }
    allK.push(k);
    allV.push(v);
    const seqLen = kvLen + 1;

    // Multi-head attention
    const x_attn = new Int32Array(n_embd);
    const scale = t.val(head_dim ** -0.5);

    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim;
      const attnLogits = new Int32Array(seqLen);
      for (let s = 0; s < seqLen; s++) {
        let dot = t.val(0);
        for (let j = 0; j < head_dim; j++) {
          dot = t_add(t, dot, t_mul(t, q[hs + j], allK[s][hs + j]));
        }
        attnLogits[s] = t_mul(t, dot, scale);
      }
      const probs = t_softmax(t, attnLogits, seqLen);
      for (let j = 0; j < head_dim; j++) {
        let sumV = t.val(0);
        for (let s = 0; s < seqLen; s++) {
          sumV = t_add(t, sumV, t_mul(t, probs[s], allV[s][hs + j]));
        }
        x_attn[hs + j] = sumV;
      }
    }

    const attnOut = t_linear(t, x_attn, params.wo, n_embd, n_embd);
    const xr = new Int32Array(n_embd);
    for (let i = 0; i < n_embd; i++) xr[i] = t_add(t, x[i], attnOut[i]);

    const xn2 = t_rmsnorm(t, xr, n_embd);
    const mlpHid = t_linear(t, xn2, params.f1, cfg.mlp_mult * n_embd, n_embd);
    const mlpAct = new Int32Array(mlpHid.length);
    for (let i = 0; i < mlpHid.length; i++) mlpAct[i] = t_relu(t, mlpHid[i]);
    const mlpOut = t_linear(t, mlpAct, params.f2, n_embd, cfg.mlp_mult * n_embd);
    x = new Int32Array(n_embd);
    for (let i = 0; i < n_embd; i++) x[i] = t_add(t, xr[i], mlpOut[i]);
  }

  // 4. Output heads
  const policy = t_linear(t, x, params.w_policy, cfg.n_actions, n_embd);
  const valArr = t_linear(t, x, params.w_value, 1, n_embd);

  // 5. Extract K/V floats for cache
  const newK = new Float64Array(cfg.n_layer * n_embd);
  const newV = new Float64Array(cfg.n_layer * n_embd);
  for (let i = 0; i < n_embd; i++) {
    newK[i] = t.data[layerK[i]];
    newV[i] = t.data[layerV[i]];
  }

  return { policy, value: valArr[0], newK, newV };
}

// ── Parameter initialization ──

function gptInitParams(tape, cfg) {
  const { n_embd, n_head, head_dim, n_layer, block_size, feat_dim, n_actions, mlp_mult } = cfg;

  function gauss() { return (Math.random() + Math.random() + Math.random() + Math.random() +
    Math.random() + Math.random() - 3) * 0.08 * (2 / Math.sqrt(6)); }
  // ~= gauss(0, 0.08) via CLT

  function initBlock(count) {
    const base = tape.size;
    for (let i = 0; i < count; i++) tape.val(gauss());
    return base;
  }

  const params = {
    w_in:     initBlock(feat_dim * n_embd),       // 48×32 = 1536
    wpe:      initBlock(block_size * n_embd),      // 16×32 = 512
    wq:       initBlock(n_embd * n_embd),          // 32×32 = 1024
    wk:       initBlock(n_embd * n_embd),          // 1024
    wv:       initBlock(n_embd * n_embd),          // 1024
    wo:       initBlock(n_embd * n_embd),          // 1024
    f1:       initBlock(mlp_mult * n_embd * n_embd), // 128×32 = 4096
    f2:       initBlock(n_embd * mlp_mult * n_embd), // 32×128 = 4096
    w_policy: initBlock(n_actions * n_embd),       // 8×32 = 256
    w_value:  initBlock(1 * n_embd),               // 32
  };

  const WEIGHTS_END = tape.size;

  // Collect all param indices for Adam
  const paramIdx = new Int32Array(WEIGHTS_END);
  for (let i = 0; i < WEIGHTS_END; i++) paramIdx[i] = i;

  return { params, WEIGHTS_END, paramIdx };
}

// ── Adam optimizer ──

function adamUpdate(tape, paramIdx, m, v, step, lr) {
  const b1 = 0.85, b2 = 0.99, eps = 1e-8;
  const b1c = 1 - b1 ** (step + 1);
  const b2c = 1 - b2 ** (step + 1);
  for (let i = 0; i < paramIdx.length; i++) {
    const pi = paramIdx[i];
    const g = tape.grad[pi];
    m[i] = b1 * m[i] + (1 - b1) * g;
    v[i] = b2 * v[i] + (1 - b2) * g * g;
    const mh = m[i] / b1c;
    const vh = v[i] / b2c;
    tape.data[pi] -= lr * mh / (Math.sqrt(vh) + eps);
  }
}
