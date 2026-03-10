'use strict';

// ══════════════════════════════════════════════
// TROOPER SPRITES
// ══════════════════════════════════════════════
const TRO_SPRITES = {};
const TRO_NAMES = [
 'trooa1','trooa2a8','trooa3a7','trooa4a6','trooa5',
 'troob1','troob2b8','troob3b7','troob4b6','troob5',
 'trooc1','trooc2c8','trooc3c7','trooc4c6','trooc5',
 'trood1','trood2d8','trood3d7','trood4d6','trood5',
 'trooe1','trooe2e8','trooe3e7','trooe4e6','trooe5',
 'troof1','troof2f8','troof3f7','troof4f6','troof5',
 'troog1','troog2g8','troog3g7','troog4g6','troog5',
 'trooh1','trooh2h8','trooh3h7','trooh4h6','trooh5',
 'trooi0','trooj0','trook0','trool0','troom0','troon0','trooo0','troop0',
 'trooq0','troor0','troos0','troot0','troou0',
];
const TRO_DEATH_FRAME_MS = 100;
const WALK_FRAMES        = ['a','b','c','d'];
const ATK_FRAMES         = ['e','f','g','h'];
const DEATH_FRAMES_NORM  = ['j','k','l','m'];         // normal (shotgun)
const DEATH_FRAMES_GIB   = ['n','o','p','q','r','s','t','u']; // rocket gib
const WALK_FRAME_MS = 140;
const ATK_FRAME_MS  = 90;

let troLoaded = 0;

for (const n of TRO_NAMES) {
  const img = new Image();
  img.src = `tro/${n}.png`;
  img.onload = () => troLoaded++;
  TRO_SPRITES[n] = img;
}

// Returns the sprite filename for (frameLetter, dir 0-7)
// dir 0=front, 4=back, 1-3=right side, 5-7=left side (mirrored)
function tooSprName(frame, dir) {
  switch (dir) {
    case 0: return `troo${frame}1`;
    case 1: case 7: return `troo${frame}2${frame}8`;
    case 2: case 6: return `troo${frame}3${frame}7`;
    case 3: case 5: return `troo${frame}4${frame}6`;
    case 4: return `troo${frame}5`;
  }
}

function getEnemySprite(enemy) {
  if (enemy.dead) {
    const now = performance.now();
    const deathSeq = enemy.gibDeath ? DEATH_FRAMES_GIB : DEATH_FRAMES_NORM;
    if (now - enemy.spr_seq_time >= TRO_DEATH_FRAME_MS) {
      enemy.spr_seq_time = now;
      if (enemy.spr_seq < deathSeq.length) enemy.spr_seq++;
    }
    return { sprite: TRO_SPRITES[`troo${deathSeq[enemy.spr_seq - 1]}0`], mirror: false };
  }

  // Direction: angle from enemy facing to player
  let rel = Math.atan2(pl.y - enemy.y, pl.x - enemy.x) - enemy.a;
  while (rel < 0) rel += Math.PI * 2;
  while (rel > Math.PI * 2) rel -= Math.PI * 2;
  const dir = Math.floor((rel / (Math.PI * 2)) * 8) & 7;
  const mirror = dir >= 5; // dirs 5,6,7 are mirrored left-side views

  // Attack frame advancement is handled in stepEnemy; just read current state here
  let frameLetter;
  if (enemy.isAttacking) {
    frameLetter = ATK_FRAMES[enemy.atkFrame];
  } else {
    const now = performance.now();
    if (enemy.walkDir !== 0 && now - enemy.walkFrameTime >= WALK_FRAME_MS) {
      enemy.walkFrameTime = now;
      enemy.walkFrame = (enemy.walkFrame + enemy.walkDir + WALK_FRAMES.length) % WALK_FRAMES.length;
    }
    frameLetter = WALK_FRAMES[enemy.walkFrame];
  }

  const name = tooSprName(frameLetter, dir);
  return { sprite: TRO_SPRITES[name] || TRO_SPRITES['trooa1'], mirror };
}

function drawSprite(sprite, spr_x, spr_y, mirror) {
  if (!sprite || !sprite.complete) return;

  const dx = spr_x - pl.x;
  const dy = spr_y - pl.y;

  let da = Math.atan2(dy, dx) - pl.a;
  while (da > Math.PI) da -= 2*Math.PI;
  while (da < -Math.PI) da += 2*Math.PI;

  if (Math.abs(da) > FOV*0.65) return;

  const dist = Math.sqrt(dx*dx + dy*dy);

  const ray = castRay(pl.x, pl.y, Math.atan2(dy,dx));
  if (ray.d < dist-0.4) return;

  // Scale relative to reference standing frame so corpse/attack frames
  // render at their natural proportional size (fixes death-anim zoom)
  const ref = TRO_SPRITES['trooa1'];
  const refH = (ref && ref.height) ? ref.height : sprite.height;
  const sh_ref = Math.min(CH, (CH / dist) | 0);
  const sh = Math.min(CH, (sh_ref * sprite.height / refH) | 0);
  if (sh <= 0) return;
  const sw = ((sh * sprite.width) / sprite.height) | 0;
  if (sw <= 0) return;

  const sx = (CW/2 + (da/(FOV/2))*(CW/2))|0;
  const x0 = sx - (sw>>1);
  // Anchor sprite bottom to floor level (bottom of wall at same distance)
  const y0 = HALF_CH + (sh_ref>>1) - sh;

  ctx.save();
  ctx.globalAlpha = Math.min(1, 1 - dist/18);

  // Z-buffer clipping per column
  for (let x=0; x<sw; x++) {
    const col = x0 + x;
    if (col<0 || col>=CW) continue;
    if (dist > zbuf[col]) continue;

    const srcX = mirror
      ? ((sw - 1 - x) * sprite.width / sw) | 0
      : (x * sprite.width / sw) | 0;

    ctx.drawImage(sprite, srcX, 0, 1, sprite.height, col, y0, 1, sh);
  }

  ctx.restore();
}

// ══════════════════════════════════════════════
// CANVAS
// ══════════════════════════════════════════════
const cv  = document.getElementById('c');
const ctx = cv.getContext('2d');
const CW = cv.width, CH = cv.height, HALF_CH = CH >> 1;
const FOV = Math.PI / 3.0;
const imgd = ctx.createImageData(CW, CH);
const px = imgd.data;
const setpx = (x, y, r, g, b) => {
  const i = (y * CW + x) << 2;
  px[i] = r; px[i+1] = g; px[i+2] = b; px[i+3] = 255;
};

// ══════════════════════════════════════════════
// GAME STATE
// ══════════════════════════════════════════════
const pl = { x:18, y:28, a:-2, hp:100, maxHp:100, rcd:0, invincible:0 };
let en = { x:13.5, y:13.5, a:Math.PI, hp:10, maxHp:20, fcd:0, dead:false, spr_seq:0, spr_seq_time:0, respT:0, flashT:0,
           walkFrame:0, walkFrameTime:0, atkFrame:0, atkFrameTime:0, isAttacking:false, walkDir:0, gibDeath:false };
const enemies = [en];
const keys = {};
window.addEventListener('keydown', e => { keys[e.code]=true;  e.preventDefault(); });
window.addEventListener('keyup',   e => { keys[e.code]=false; });
const gameOverBanner = document.getElementById('game-over-banner');
const restartButton = document.getElementById('banner-restart');
let gameOverActive = false;
const triggerGameOver = () => {
  if (gameOverActive) return;
  gameOverActive = true;
  gameOverBanner.classList.add('visible');
};

// ══════════════════════════════════════════════
// COLLISION
// ══════════════════════════════════════════════
const R = 0.25;
const collidesAt = (x, y) =>
  mapAt(x-R,y-R)||mapAt(x+R,y-R)||mapAt(x-R,y+R)||mapAt(x+R,y+R)||
  mapAt(x,y-R)  ||mapAt(x,y+R)  ||mapAt(x-R,y)  ||mapAt(x+R,y);

// ══════════════════════════════════════════════
// PROJECTILES
// ══════════════════════════════════════════════
const projs = [];
const exps  = [];
let kills = 0;

const RKT_COOLDOWN = 60;
function spawnRocket() {
  projs.push({
    x: pl.x+Math.cos(pl.a)*0.55, y: pl.y+Math.sin(pl.a)*0.55,
    a: pl.a, spd: 0.1, type:'rocket', alive:true, life:0, maxLife:130
  });
  pl.rcd = RKT_COOLDOWN;
}

// ══════════════════════════════════════════════
// RAYCASTER — DDA
// ══════════════════════════════════════════════
const zbuf = new Float32Array(CW);

function castRay(ox, oy, angle) {
  const rdx=Math.cos(angle), rdy=Math.sin(angle);
  let mx=ox|0, my=oy|0;
  const sx=rdx>0?1:-1, sy=rdy>0?1:-1;
  const ddx=Math.abs(1/(rdx||1e-12)), ddy=Math.abs(1/(rdy||1e-12));
  let sdx=(rdx>0?mx+1-ox:ox-mx)*ddx, sdy=(rdy>0?my+1-oy:oy-my)*ddy;
  let side=0;
  for (let i=0; i<64; i++) {
    if (sdx<sdy) { sdx+=ddx; mx+=sx; side=0; }
    else         { sdy+=ddy; my+=sy; side=1; }
    if (MAP[my*MW+mx]) return { d: side===0?sdx-ddx:sdy-ddy, side };
  }
  return { d:64, side:0 };
}

function drawScene() {
  for (let y=0; y<CH; y++) {
    if (y<HALF_CH) {
      const t=y/HALF_CH, r=(4+t*14)|0;
      for (let x=0; x<CW; x++) setpx(x,y,r,0,0);
    } else {
      const t=(y-HALF_CH)/HALF_CH, r=(18+t*14)|0, g=(t*5)|0;
      for (let x=0; x<CW; x++) setpx(x,y,r,g,0);
    }
  }
  for (let col=0; col<CW; col++) {
    const ra = pl.a - FOV*0.5 + (col/CW)*FOV;
    const { d, side } = castRay(pl.x, pl.y, ra);
    const corr = d*Math.cos(ra-pl.a);
    zbuf[col] = corr;
    const lh = Math.min(CH, ((CH/(corr+0.001))+0.5)|0);
    const y0 = Math.max(0, (HALF_CH-lh/2)|0);
    const y1 = Math.min(CH-1, (HALF_CH+lh/2)|0);
    const br = Math.max(0, Math.min(255, (200/(corr+0.1))|0));
    const r = (br*(side?0.5:1.0))|0, g=(r*0.06)|0;
    for (let y=y0; y<=y1; y++) setpx(col,y,r,g,0);
  }
  ctx.putImageData(imgd,0,0);
}

// ══════════════════════════════════════════════
// SPRITE HELPERS
// ══════════════════════════════════════════════
function projectSprite(wx, wy) {
  const dx=wx-pl.x, dy=wy-pl.y;
  const dist=Math.sqrt(dx*dx+dy*dy);
  if (dist<0.12) return null;
  let da=Math.atan2(dy,dx)-pl.a;
  while (da> Math.PI) da-=2*Math.PI;
  while (da<-Math.PI) da+=2*Math.PI;
  if (Math.abs(da)>FOV*0.72) return null;
  return { sx:(CW/2+(da/(FOV/2))*(CW/2))|0, dist, da };
}

function hasLOS(x1, y1, x2, y2) {
  const { d } = castRay(x1, y1, Math.atan2(y2-y1, x2-x1));
  return d >= Math.hypot(x2-x1, y2-y1)-0.4;
}

// ══════════════════════════════════════════════
// DRAW ENEMY
// ══════════════════════════════════════════════
function drawEnemy() {
  // Sort farthest first (painter's algorithm)
  const sorted = enemies.slice().sort((a, b) =>
    Math.hypot(b.x-pl.x, b.y-pl.y) - Math.hypot(a.x-pl.x, a.y-pl.y)
  );
  for (const enemy of sorted) {
    const { sprite, mirror } = getEnemySprite(enemy);
    if (sprite) drawSprite(sprite, enemy.x, enemy.y, mirror);
  }
}

// ══════════════════════════════════════════════
// DRAW PROJECTILES
// ══════════════════════════════════════════════
function drawProjectiles() {
  for (const p of projs) {
    if (!p.alive) continue;
    const wdist=Math.hypot(p.x-pl.x, p.y-pl.y);
    const { d } = castRay(pl.x, pl.y, Math.atan2(p.y-pl.y, p.x-pl.x));
    if (d < wdist-0.25) continue;
    const s=projectSprite(p.x, p.y);
    if (!s) continue;
    const { sx, dist } = s;
    if (sx<0||sx>=CW) continue;
    const sz=Math.max(3,(18/dist)|0);
    ctx.save();
    if (p.type==='rocket') {
      ctx.shadowColor='#aaddff'; ctx.shadowBlur=14;
      ctx.fillStyle='#ffffff';
      ctx.beginPath(); ctx.arc(sx,HALF_CH,sz,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#88bbff'; ctx.shadowBlur=0;
      ctx.beginPath(); ctx.arc(sx,HALF_CH,sz*0.5,0,Math.PI*2); ctx.fill();
    } else {
      ctx.shadowColor='#ff3300'; ctx.shadowBlur=16;
      ctx.fillStyle='#ff6600';
      ctx.beginPath(); ctx.arc(sx,HALF_CH,sz*1.4,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#ffcc00'; ctx.shadowBlur=0;
      ctx.beginPath(); ctx.arc(sx,HALF_CH,sz*0.55,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();
  }
}

// ══════════════════════════════════════════════
// DRAW EXPLOSIONS
// ══════════════════════════════════════════════
function drawExplosions() {
  for (const ex of exps) {
    const s=projectSprite(ex.x, ex.y);
    if (!s) continue;
    const { sx, dist } = s;
    const progress=1-ex.t/ex.maxT;
    const radius=Math.max(4,(CH*0.28*progress/(dist+0.3))|0);
    const isHit=ex.type==='hit';
    ctx.save();
    ctx.globalAlpha=(ex.t/ex.maxT)*0.88;
    ctx.shadowColor=isHit?'#ffaa00':'#ff4400'; ctx.shadowBlur=20;
    ctx.strokeStyle=isHit?'#ffaa00':'#ff5500';
    ctx.fillStyle  =isHit?'rgba(255,160,0,0.2)':'rgba(255,80,0,0.2)';
    ctx.lineWidth=2.5;
    ctx.beginPath(); ctx.arc(sx,HALF_CH,radius,0,Math.PI*2);
    ctx.stroke(); ctx.fill();
    ctx.restore();
  }
}

// ══════════════════════════════════════════════
// MINIMAP
// ══════════════════════════════════════════════
function drawMinimap() {
  const S=5, ox=6, oy=6;
  ctx.save();
  ctx.fillStyle='rgba(0,0,0,0.72)';
  ctx.fillRect(ox-1,oy-1,MW*S+2,MH*S+2);
  for (let my=0; my<MH; my++) for (let mx=0; mx<MW; mx++) {
    ctx.fillStyle=MAP[my*MW+mx]?'#b0b0b0':'#180808';
    ctx.fillRect(ox+mx*S,oy+my*S,S-1,S-1);
  }
  for (const p of projs) {
    if (!p.alive) continue;
    ctx.fillStyle=p.type==='rocket'?'#88ccff':'#ff8800';
    ctx.fillRect((ox+p.x*S-1)|0,(oy+p.y*S-1)|0,2,2);
  }
  ctx.fillStyle='#44ff66';
  ctx.fillRect((ox+pl.x*S-2)|0,(oy+pl.y*S-2)|0,4,4);
  ctx.strokeStyle='#44ff66'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(ox+pl.x*S,oy+pl.y*S);
  ctx.lineTo(ox+(pl.x+Math.cos(pl.a)*2)*S,oy+(pl.y+Math.sin(pl.a)*2)*S);
  ctx.stroke();
  for (const enemy of enemies) {
    if (enemy.dead) {
      ctx.fillStyle='#505050';
      ctx.fillRect((ox+enemy.x*S-1)|0,(oy+enemy.y*S-1)|0,3,3);
    } else {
      ctx.fillStyle='#ff2222'; ctx.shadowColor='#ff2222'; ctx.shadowBlur=4;
      ctx.fillRect((ox+enemy.x*S-2)|0,(oy+enemy.y*S-2)|0,4,4);
      ctx.shadowBlur=0;
    }
  }
  ctx.restore();
}

function drawRedFlicker() {
    ctx.save();
    ctx.fillStyle='rgba(255,60,0,0.06)';
    ctx.fillRect(0,0,CW,CH);
    ctx.restore();
}

function resetGameState() {
  gameOverActive = false;
  gameOverBanner.classList.remove('visible');
  pl.x = 18; pl.y = 28; pl.a = -2; pl.hp = pl.maxHp; pl.rcd = 0; pl.invincible = 0;
  en = { x:13.5, y:13.5, a:Math.PI, hp:10, maxHp:20, fcd:0, dead:false, spr_seq:0, spr_seq_time:0, respT:0, flashT:0,
         walkFrame:0, walkFrameTime:0, atkFrame:0, atkFrameTime:0, isAttacking:false, walkDir:0, gibDeath:false };
  enemies.length = 0;
  enemies.push(en);
  projs.length = 0;
  exps.length = 0;
  kills = 0;
  steps = 0;
  curLoss = 1.0;
  lossHist.length = 0;
  lastOut.fill(0);
  dmgFlashT = 0;
  dmgFlash.className = '';
  Object.keys(keys).forEach(code => keys[code] = false);
  updateHUD();
}

restartButton.addEventListener('click', resetGameState);

// ══════════════════════════════════════════════
// LOSS GRAPH
// ══════════════════════════════════════════════
const lc=document.getElementById('loss-canvas');
const lctx=lc.getContext('2d');
const LW=lc.width, LH=lc.height;

function drawLoss() {
  lctx.fillStyle='#040202';
  lctx.fillRect(0,0,LW,LH);
  if (lossHist.length<3) return;

  lctx.strokeStyle='#cc1a1a18'; lctx.lineWidth=1;
  for (let g=1; g<4; g++) {
    lctx.beginPath();
    lctx.moveTo(0,(g/4)*LH); lctx.lineTo(LW,(g/4)*LH);
    lctx.stroke();
  }

  const maxL=Math.max(...lossHist,0.02);
  const minL=Math.min(...lossHist,0);
  const range=maxL-minL||1;
  const n=lossHist.length;

  lctx.beginPath();
  for (let i=0; i<n; i++) {
    const x=(i/(n-1))*LW;
    const y=LH-((lossHist[i]-minL)/range)*(LH-4)-2;
    i===0?lctx.moveTo(x,y):lctx.lineTo(x,y);
  }
  lctx.lineTo(LW,LH); lctx.lineTo(0,LH); lctx.closePath();
  lctx.fillStyle='rgba(180,20,20,0.08)'; lctx.fill();

  lctx.beginPath();
  for (let i=0; i<n; i++) {
    const x=(i/(n-1))*LW;
    const y=LH-((lossHist[i]-minL)/range)*(LH-4)-2;
    i===0?lctx.moveTo(x,y):lctx.lineTo(x,y);
  }
  lctx.strokeStyle='#cc1a1a'; lctx.lineWidth=1.5;
  lctx.shadowColor='#cc1a1a'; lctx.shadowBlur=5;
  lctx.stroke(); lctx.shadowBlur=0;
}

// ══════════════════════════════════════════════
// HUD UPDATE
// ══════════════════════════════════════════════
const PHASES=[
  [0,    'RANDOM MOVEMENT...'],
  [400,  'SENSING DIRECTION...'],
  [1000, 'TRACKING PLAYER...'],
  [2000, '► HUNTING MODE ACTIVE'],
  [3500, '■ FULL PREDATOR LOCK'],
  [6000, '▲ DODGE + FIRE LEARNED'],
];

// Build output buttons once
const outGrid=document.getElementById('out-grid');
const outBtns=[];
OUTPUT_LABELS.forEach((name,i)=>{
  const b=document.createElement('div');
  b.className='out-btn'; b.textContent=name;
  outGrid.appendChild(b); outBtns.push(b);
});

// Build NN arch display
document.getElementById('nn-arch-box').innerHTML=
  `<span>INPUT</span>  [${INPUT_LABELS.join(', ')}]<br>`+
  `<span>LAYER 1</span> ${HIDDEN_LAYER_1_SIZE} neurons · ReLU<br>`+
  `<span>LAYER 2</span> ${HIDDEN_LAYER_2_SIZE} neurons · ReLU<br>`+
  `<span>OUTPUT</span> [${OUTPUT_LABELS.join(', ')}]<br>`+
  `<span>ACT</span>   Sigmoid · threshold 0.5<br>`+
  `<span>OPT</span>   Adam · lr=${LEARNING_RATE}<br>`+
  `<span>LOSS</span>  MSE vs teacher policy`;

// Rocket cooldown bar
const rcdFill=document.getElementById('rcd-fill');

function updateHUD() {
  // HP bars
  document.getElementById('hp-pl').style.width=(pl.hp/pl.maxHp*100)+'%';
  document.getElementById('hv-pl').textContent=pl.hp;
  document.getElementById('hp-en').style.width=(en.hp/en.maxHp*100)+'%';
  document.getElementById('hv-en').textContent=en.dead?'DEAD':en.hp;

  // Stats
  document.getElementById('s-steps').textContent=steps;
  document.getElementById('s-loss').textContent=curLoss.toFixed(3);
  document.getElementById('s-kills').textContent=kills;

  // Phase
  const prog=Math.min(1,steps/6000);
  document.getElementById('pf').style.width=(prog*100)+'%';
  let lbl=PHASES[0][1];
  for (const [thr,name] of PHASES) { if (steps>=thr) lbl=name; }
  document.getElementById('pl-txt').textContent=lbl;

  // Output indicators
  outBtns.forEach((b,i)=>{
    const on=lastOut[i]>0.5;
    if (i===6) b.className='out-btn'+(on?' fire-on':'');
    else       b.className='out-btn'+(on?' on':'');
  });

  // Rocket cooldown
  rcdFill.style.width=((1-pl.rcd/RKT_COOLDOWN)*100)+'%';
}
