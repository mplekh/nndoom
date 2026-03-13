'use strict';

// ══════════════════════════════════════════════
// MICROGPT CONTROLLER — Game integration + A2C
// ══════════════════════════════════════════════

// Modes
const SNN_MODES = ['OBSERVE', 'ASSISTIVE', 'AUTONOMOUS'];
let snnMode = 1;  // start in ASSISTIVE
let snnActive = false;
let snnShowRaster = true;
let snnFrameCount = 0;
let snnTotalReward = 0;

// Reward tracking
let prevKills = 0;
let prevPlHp  = 100;
let prevEnHp  = 100;
let prevPlX = 0, prevPlY = 0;
let prevEnDist = 10;
let prevAimErr = Math.PI;  // previous absolute angle error to enemy

// 4 axis outputs:
//  0: move_fb   (tanh: >0 forward, <0 backward)
//  1: turn_lr   (tanh: >0 turn right, <0 turn left)
//  2: strafe_lr (tanh: >0 strafe right, <0 strafe left)
//  3: fire      (sigmoid: >0.5 fire)
const AXIS_NAMES = ['MOVE', 'TURN', 'STRAFE', 'FIRE'];
const AXIS_DEAD = 0.2;  // dead zone for axes

// Visual config
const VIS_W = 16, VIS_H = 12;
const VIS_PIXELS = VIS_W * VIS_H;  // 192
const VIS_BLOCK_W = 50, VIS_BLOCK_H = 50;  // 800/16, 600/12

// Previous frame for temporal differencing (unused — transformer uses KV cache)
let prevGray = new Float64Array(VIS_PIXELS);
let visGray  = new Float64Array(VIS_PIXELS);   // for HUD
let visEvents = new Int8Array(VIS_PIXELS);

// Tape + model state
let tape, gptParams, WEIGHTS_END, paramIdx;
let adamM, adamV;

// KV cache (circular buffer)
let kvKeys, kvVals;
let kvLen = 0, kvPos = 0;

// Training state
let prevValue = 0;
let prevAction = 0;
let polLoss = 0;
let gptStep = 0;

// Action outputs for visualization (last frame): raw axis values [-1,1] or [0,1]
let lastActionProbs = new Float64Array(4);

// Game state feature count
const GAME_FEAT = 16;
const FEAT_DIM = GPT.feat_dim;  // 208 = 192 visual + 16 game state

// ── Init ────────────────────────────────────
function gptCtrlInit() {
  tape = new Tape();
  const init = gptInitParams(tape, GPT);
  gptParams = init.params;
  WEIGHTS_END = init.WEIGHTS_END;
  paramIdx = init.paramIdx;

  adamM = new Float64Array(paramIdx.length);
  adamV = new Float64Array(paramIdx.length);

  const cacheSize = GPT.n_layer * GPT.block_size * GPT.n_embd;
  kvKeys = new Float64Array(cacheSize);
  kvVals = new Float64Array(cacheSize);
  kvLen = 0;
  kvPos = 0;

  prevGray = new Float64Array(VIS_PIXELS);
  prevValue = 0;
  prevAction = 0;
  gptStep = 0;

  snnActive = true;
  prevKills = kills;
  prevPlHp  = pl.hp;
  prevPlX   = pl.x;
  prevPlY   = pl.y;
  prevEnDist = Math.hypot(en.x - pl.x, en.y - pl.y);
  prevAimErr = Math.PI;
  snnFrameCount = 0;
  snnTotalReward = 0;

  console.log(`GPT Controller: ${WEIGHTS_END} params, ${GPT.n_layer}L/${GPT.n_head}H/${GPT.n_embd}D, KV cache ${GPT.block_size}`);
}

// ── Visual downsampling (8×4) ────────────────
function downsampleVis(pxData) {
  const out = new Float64Array(VIS_PIXELS);
  for (let by = 0; by < VIS_H; by++) {
    const baseY = by * VIS_BLOCK_H;
    for (let bx = 0; bx < VIS_W; bx++) {
      const baseX = bx * VIS_BLOCK_W;
      let sum = 0;
      // Sample 5×5 grid within each 100×150 block
      for (let sy = 0; sy < 5; sy++) {
        const py = baseY + ((sy * VIS_BLOCK_H / 5) | 0);
        for (let sx = 0; sx < 5; sx++) {
          const px = baseX + ((sx * VIS_BLOCK_W / 5) | 0);
          const i = (py * 800 + px) << 2;
          // BT.709 luma
          sum += pxData[i] * 0.2126 + pxData[i + 1] * 0.7152 + pxData[i + 2] * 0.0722;
        }
      }
      out[by * VIS_W + bx] = sum / (25 * 255);  // normalize to [0,1]
    }
  }
  return out;
}

// ── Game state extraction ───────────────────
function extractGameState(player, enemy, projectiles) {
  const gs = new Float64Array(GAME_FEAT);
  const MAX_D = 20;

  // 0: player hp
  gs[0] = player.hp / player.maxHp;

  // 1-2: player angle sin/cos
  gs[1] = Math.sin(player.a);
  gs[2] = Math.cos(player.a);

  // 3-4: enemy relative position
  const dx = enemy.x - player.x;
  const dy = enemy.y - player.y;
  gs[3] = dx / MAX_D;
  gs[4] = dy / MAX_D;

  // 5: enemy distance
  const dist = Math.hypot(dx, dy);
  gs[5] = dist / MAX_D;

  // 6-7: relative angle to enemy sin/cos
  const angleToEnemy = Math.atan2(dy, dx) - player.a;
  gs[6] = Math.sin(angleToEnemy);
  gs[7] = Math.cos(angleToEnemy);

  // 8: LOS
  gs[8] = (!enemy.dead && hasLOS(player.x, player.y, enemy.x, enemy.y)) ? 1 : 0;

  // 9: enemy hp
  gs[9] = enemy.dead ? 0 : enemy.hp / enemy.maxHp;

  // 10: weapon cooldown
  gs[10] = player.rcd / (player.maxRcd || 22);

  // 11-14: nearest projectile (enemy fireball)
  let nearDist = MAX_D;
  let nearRX = 0, nearRY = 0, nearApproach = 0;
  for (const p of projectiles) {
    if (!p.alive || p.type !== 'fireball') continue;
    const pdx = p.x - player.x;
    const pdy = p.y - player.y;
    const pd = Math.hypot(pdx, pdy);
    if (pd < nearDist) {
      nearDist = pd;
      nearRX = pdx / MAX_D;
      nearRY = pdy / MAX_D;
      // Is it approaching?
      const vx = Math.cos(p.a) * p.spd;
      const vy = Math.sin(p.a) * p.spd;
      nearApproach = (pdx * vx + pdy * vy) < 0 ? 1 : 0;
    }
  }
  gs[11] = nearRX;
  gs[12] = nearRY;
  gs[13] = nearDist / MAX_D;
  gs[14] = nearApproach;

  // 15: previous action
  gs[15] = prevAction / 7;

  return gs;
}

// ── Reward computation ──────────────────────
function computeReward(player, enemy, projectiles, aiKeys) {
  let reward = 0;

  const dx = enemy.x - player.x;
  const dy = enemy.y - player.y;
  const enDist = Math.hypot(dx, dy);
  const angleToEnemy = Math.atan2(dy, dx) - player.a;
  const aimErr = Math.abs(Math.atan2(Math.sin(angleToEnemy), Math.cos(angleToEnemy)));
  const los = !enemy.dead && hasLOS(player.x, player.y, enemy.x, enemy.y);
  const aimed = aimErr < 0.15;

  // Kill — big reward
  if (kills > prevKills) { reward += 5.0; prevKills = kills; }

  // Hit enemy — strong reward
  if (!enemy.dead && enemy.hp < prevEnHp) reward += 3.0;
  prevEnHp = enemy.dead ? enemy.maxHp : enemy.hp;

  // AIM IMPROVEMENT — the key anti-orbit signal
  // Reward REDUCING angle error, penalize INCREASING it
  if (!enemy.dead) {
    const aimImprovement = prevAimErr - aimErr;  // positive = getting more aimed
    reward += aimImprovement * 1.5;
  }
  prevAimErr = aimErr;

  // Small bonus for being well-aimed (not just improving)
  if (los && aimed) reward += 0.05;

  // Firing while aimed — immediate reward
  if (aiKeys['Space'] && los && aimed) reward += 1.0;
  // Firing while NOT aimed — penalize
  else if (aiKeys['Space'] && !aimed) reward -= 0.2;

  // Approaching enemy
  if (!enemy.dead) {
    reward += (prevEnDist - enDist) * 0.3;
  }
  prevEnDist = enDist;

  // Took damage
  if (player.hp < prevPlHp) reward -= 1.0;
  prevPlHp = player.hp;

  // Death
  if (player.hp <= 0) reward -= 5.0;

  // Wall bumping
  const moved = Math.hypot(player.x - prevPlX, player.y - prevPlY);
  const tryingToMove = aiKeys['KeyW'] || aiKeys['KeyS'] || aiKeys['KeyA'] || aiKeys['KeyD'];
  if (tryingToMove && moved < 0.005) reward -= 0.2;
  prevPlX = player.x;
  prevPlY = player.y;

  // Time penalty
  reward -= 0.005;

  return reward;
}

// ── Ideal action heuristic (teacher) ────────
function computeIdealActions(player, enemy, projectiles) {
  const ideal = new Float64Array(4);  // move, turn, strafe, fire

  const dx = enemy.x - player.x;
  const dy = enemy.y - player.y;
  const dist = Math.hypot(dx, dy);
  const los = !enemy.dead && hasLOS(player.x, player.y, enemy.x, enemy.y);

  // Use pathfinding when no LOS to navigate around walls
  let navDx = dx, navDy = dy;
  if (!los && !enemy.dead) {
    const nav = pathDir(player.x, player.y, enemy.x, enemy.y);
    navDx = nav.dx; navDy = nav.dy;
  }

  const angleToNav = Math.atan2(navDy, navDx) - player.a;
  const aimErr = Math.atan2(Math.sin(angleToNav), Math.cos(angleToNav));  // signed
  const absAim = Math.abs(aimErr);

  // Turn toward navigation target (enemy or pathfinding waypoint)
  ideal[1] = Math.max(-1, Math.min(1, aimErr * 3));

  // Move forward when roughly facing target, back up if too close
  if (enemy.dead) {
    ideal[0] = 0.3;  // wander forward
  } else if (dist < 2 && los) {
    ideal[0] = -0.5;  // too close with LOS, back up
  } else if (absAim < 0.8) {
    ideal[0] = 0.7;   // facing roughly right, approach
  } else {
    ideal[0] = 0.1;   // not facing target, slow movement
  }

  // Strafe to dodge incoming fireballs
  let nearFB = null, nearDist = 5;
  for (const p of projectiles) {
    if (!p.alive || p.type !== 'fireball') continue;
    const pd = Math.hypot(p.x - player.x, p.y - player.y);
    if (pd < nearDist) { nearDist = pd; nearFB = p; }
  }
  if (nearFB && nearDist < 3) {
    const fbAngle = nearFB.a - player.a;
    const cross = Math.sin(fbAngle);
    ideal[2] = cross > 0 ? -0.8 : 0.8;
  }

  // Fire only when aimed AND in LOS (not through walls)
  if (los && absAim < 0.2 && player.rcd <= 0) {
    ideal[3] = 1.0;
  } else {
    ideal[3] = -1.0;
  }

  return ideal;
}

// ── Key blending ────────────────────────────
function blendKeys(humanKeys, aiKeys) {
  const mode = SNN_MODES[snnMode];
  if (mode === 'OBSERVE') return humanKeys;
  if (mode === 'ASSISTIVE') {
    const eff = {};
    for (const k in humanKeys) if (humanKeys[k]) eff[k] = true;
    for (const k in aiKeys)   if (aiKeys[k])     eff[k] = true;
    eff['Digit1'] = humanKeys['Digit1'];
    eff['Digit2'] = humanKeys['Digit2'];
    return eff;
  }
  // AUTONOMOUS
  aiKeys['Digit1'] = humanKeys['Digit1'];
  aiKeys['Digit2'] = humanKeys['Digit2'];
  return aiKeys;
}

// ── Per-frame pipeline ──────────────────────
function snnCtrlStep(pxBuf, player, enemy, allEnemies, projectiles, humanKeys) {
  if (!snnActive) return humanKeys;
  detectGameReset(player);
  snnFrameCount++;

  // 1. Visual features (32)
  const gray = downsampleVis(pxBuf);
  for (let i = 0; i < VIS_PIXELS; i++) visGray[i] = gray[i] * 255;
  // Temporal diff for HUD visualization
  for (let i = 0; i < VIS_PIXELS; i++) {
    const d = gray[i] - prevGray[i];
    visEvents[i] = d > 0.05 ? 1 : d < -0.05 ? -1 : 0;
  }

  // 2. Game state features (16)
  const gs = extractGameState(player, enemy, projectiles);

  // 3. Reset tape to weights only
  tape.truncate(WEIGHTS_END);
  tape.zero_grad();

  // 4. Push 48 input features as leaves
  const inputIdx = new Int32Array(FEAT_DIM);
  for (let i = 0; i < VIS_PIXELS; i++) inputIdx[i] = tape.val(gray[i]);
  for (let i = 0; i < GAME_FEAT; i++) inputIdx[VIS_PIXELS + i] = tape.val(gs[i]);

  // 5. GPT forward pass
  const posId = Math.min(kvLen, GPT.block_size - 1);
  const result = gpt_forward_full(tape, inputIdx, posId,
    { keys: kvKeys, vals: kvVals }, kvLen, gptParams, GPT);

  // 6. Compute ideal actions from game state (heuristic teacher)
  const ideal = computeIdealActions(player, enemy, projectiles);

  // 7. Read model outputs
  const policyIdx = result.policy;  // Int32Array(4) tape indices
  const modelOut = new Float64Array(GPT.n_actions);
  for (let a = 0; a < GPT.n_actions; a++) modelOut[a] = tape.data[policyIdx[a]];

  // Store for HUD
  for (let a = 0; a < GPT.n_actions; a++) lastActionProbs[a] = modelOut[a];

  // 8. Store K/V in circular cache
  const cacheOff = kvPos * GPT.n_embd;
  for (let i = 0; i < GPT.n_embd; i++) {
    kvKeys[cacheOff + i] = result.newK[i];
    kvVals[cacheOff + i] = result.newV[i];
  }
  kvPos = (kvPos + 1) % GPT.block_size;
  if (kvLen < GPT.block_size) kvLen++;

  // 9. Map model outputs to keys
  const aiKeys = {};
  if (modelOut[0] >  AXIS_DEAD) aiKeys['KeyW'] = true;
  if (modelOut[0] < -AXIS_DEAD) aiKeys['KeyS'] = true;
  if (modelOut[1] >  AXIS_DEAD) aiKeys['ArrowRight'] = true;
  if (modelOut[1] < -AXIS_DEAD) aiKeys['ArrowLeft'] = true;
  if (modelOut[2] >  AXIS_DEAD) aiKeys['KeyD'] = true;
  if (modelOut[2] < -AXIS_DEAD) aiKeys['KeyA'] = true;
  if (modelOut[3] > 0) aiKeys['Space'] = true;

  // 10. Track reward for HUD
  const reward = computeReward(player, enemy, projectiles, aiKeys);
  snnTotalReward += reward;

  // 11. Supervised training: MSE loss against ideal actions
  let loss = tape.val(0);
  for (let a = 0; a < GPT.n_actions; a++) {
    const diff = t_add(tape, policyIdx[a], tape.val(-ideal[a]));
    const sq = t_mul(tape, diff, diff);
    loss = t_add(tape, loss, sq);
  }

  tape.backward(loss);
  const lr = 0.005 * Math.max(0.1, 1 - snnFrameCount / 80000);
  adamUpdate(tape, paramIdx, adamM, adamV, gptStep, lr);
  gptStep++;

  polLoss = polLoss * 0.95 + tape.data[loss] * 0.05;

  prevAction = aiKeys['Space'] ? 6 : aiKeys['KeyW'] ? 0 : 7;

  // Save previous frame
  for (let i = 0; i < VIS_PIXELS; i++) prevGray[i] = gray[i];

  return blendKeys(humanKeys, aiKeys);
}

// ── Keyboard controls ───────────────────────
let _snnKey3 = false, _snnKey4 = false;
window.addEventListener('keydown', e => {
  if (e.code === 'Digit3' && !_snnKey3) {
    _snnKey3 = true;
    snnMode = (snnMode + 1) % SNN_MODES.length;
    console.log('GPT Mode:', SNN_MODES[snnMode]);
  }
  if (e.code === 'Digit4' && !_snnKey4) {
    _snnKey4 = true;
    snnShowRaster = !snnShowRaster;
    const rc = document.getElementById('snn-raster');
    if (rc) rc.style.display = snnShowRaster ? 'block' : 'none';
  }
});
window.addEventListener('keyup', e => {
  if (e.code === 'Digit3') _snnKey3 = false;
  if (e.code === 'Digit4') _snnKey4 = false;
});

// ── Visual input visualization ──────────────
function drawVisualInput() {
  const vc = document.getElementById('snn-vis');
  if (!vc) return;
  const vctx = vc.getContext('2d');
  const id = vctx.createImageData(VIS_W, VIS_H);
  const d = id.data;

  for (let i = 0; i < VIS_PIXELS; i++) {
    const off = i << 2;
    const g = Math.min(255, visGray[i]) | 0;
    const ev = visEvents[i];
    if (ev === 1) {
      d[off] = g >> 1; d[off + 1] = Math.min(255, g + 140); d[off + 2] = g >> 2;
    } else if (ev === -1) {
      d[off] = Math.min(255, g + 140); d[off + 1] = g >> 2; d[off + 2] = g >> 2;
    } else {
      d[off] = g; d[off + 1] = g; d[off + 2] = g;
    }
    d[off + 3] = 255;
  }
  vctx.putImageData(id, 0, 0);
}

// ── Action axis visualization ────────────────

function drawAttnViz() {
  const rc = document.getElementById('snn-raster');
  if (!rc) return;
  const rctx = rc.getContext('2d');
  const rw = rc.width, rh = rc.height;

  rctx.fillStyle = '#040202';
  rctx.fillRect(0, 0, rw, rh);

  const n = GPT.n_actions;
  const barH = ((rh - 4) / n) | 0;
  const midX = 36 + ((rw - 40) >> 1);
  const halfW = (rw - 44) >> 1;

  for (let a = 0; a < n; a++) {
    const v = lastActionProbs[a];
    const y = 2 + a * barH;
    // Map unbounded value to [0,1] range for display via tanh
    const mapped = Math.tanh(v);
    const color = a === 1 ? '#ff8844' : a === 3 ? '#ffbb00' : '#44bbff';
    const active = a < 3 ? Math.abs(v) > AXIS_DEAD : v > 0;

    // Signed axis: draw from center
    const w = Math.abs(mapped) * halfW;
    rctx.fillStyle = active ? color : color + '44';
    if (mapped > 0) {
      rctx.fillRect(midX, y, w, barH - 1);
    } else {
      rctx.fillRect(midX - w, y, w, barH - 1);
    }
    // Center line
    rctx.fillStyle = '#ffffff22';
    rctx.fillRect(midX, y, 1, barH - 1);

    // Label
    rctx.fillStyle = '#44bbff88';
    rctx.font = '9px monospace';
    rctx.fillText(AXIS_NAMES[a], 1, y + barH - 2);
  }
}

// ── HUD update ──────────────────────────────
function updateSNNHUD() {
  const modeEl   = document.getElementById('snn-mode');
  const spikesEl = document.getElementById('snn-spikes');
  const lossEl   = document.getElementById('snn-loss');
  const rewardEl = document.getElementById('snn-reward');

  if (modeEl)   modeEl.textContent   = SNN_MODES[snnMode];
  if (spikesEl) spikesEl.textContent = kvLen;
  if (lossEl)   lossEl.textContent   = polLoss.toFixed(4);
  if (rewardEl) rewardEl.textContent = snnTotalReward.toFixed(1);

  drawVisualInput();
  drawAttnViz();
}

// ── Detect game reset and clear KV cache ────
let _lastPlHp = 0;
function detectGameReset(player) {
  // If player HP jumped from <=0 to full, game was restarted
  if (_lastPlHp <= 0 && player.hp >= player.maxHp) {
    kvLen = 0;
    kvPos = 0;
    prevValue = 0;
    prevPlHp = player.hp;
    prevEnHp = en.hp;
    prevKills = kills;
    prevGray.fill(0);
  }
  _lastPlHp = player.hp;
}

// ── Auto-init ───────────────────────────────
gptCtrlInit();
