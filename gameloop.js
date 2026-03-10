'use strict';

// ══════════════════════════════════════════════
// PLAYER UPDATE
// ══════════════════════════════════════════════
function updatePlayer() {
  if (gameOverActive) return;
  const MS=0.06, TS=0.038;
  const ca=Math.cos(pl.a), sa=Math.sin(pl.a);
  const sr=Math.cos(pl.a+Math.PI/2), sp=Math.sin(pl.a+Math.PI/2);
  const tryMove=(nx,ny)=>{
    if (!collidesAt(nx,pl.y)) pl.x=nx;
    if (!collidesAt(pl.x,ny)) pl.y=ny;
  };
  if (keys['KeyW'])      tryMove(pl.x+ca*MS, pl.y+sa*MS);
  if (keys['ArrowUp'])   tryMove(pl.x+ca*MS, pl.y+sa*MS);
  if (keys['KeyS'])      tryMove(pl.x-ca*MS, pl.y-sa*MS);
  if (keys['ArrowDown']) tryMove(pl.x-ca*MS, pl.y-sa*MS);
  if (keys['KeyA'])      tryMove(pl.x-sr*MS, pl.y-sp*MS);
  if (keys['KeyD'])      tryMove(pl.x+sr*MS, pl.y+sp*MS);
  if (keys['ArrowLeft']) pl.a-=TS;
  if (keys['ArrowRight'])pl.a+=TS;
  if (keys['Space'] && pl.rcd<=0) spawnRocket();
  if (pl.rcd>0) pl.rcd--;
  if (pl.invincible>0) pl.invincible--;
}

const dmgFlash = document.getElementById('dmg-flash');
let dmgFlashT = 0;

function updateProjectiles() {
  for (const p of projs) {
    if (!p.alive) continue;
    p.life++;
    if (p.life >= p.maxLife) { p.alive=false; continue; }
    p.x += Math.cos(p.a)*p.spd;
    p.y += Math.sin(p.a)*p.spd;

    if (mapAt(p.x, p.y)) {
      p.alive=false;
      exps.push({x:p.x, y:p.y, t:10, maxT:10, type:p.type});
      continue;
    }

    if (p.type==='rocket' && !en.dead) {
      if (Math.hypot(p.x-en.x, p.y-en.y) < 0.65) {
        p.alive=false;
        en.hp = Math.max(0, en.hp-5);
        en.flashT = 9;
        exps.push({x:p.x, y:p.y, t:16, maxT:16, type:'hit'});
        if (en.hp<=0) { en.dead=true; en.respT=210; kills++; en.spr_seq=0;
          en.spr_seq_time=performance.now()-TRO_DEATH_FRAME_MS;
          en.gibDeath=(p.type==='rocket'); }
      }
    }

    if (p.type==='fireball' && pl.invincible<=0) {
      if (Math.hypot(p.x-pl.x, p.y-pl.y) < 0.6) {
        p.alive=false;
        pl.hp = Math.max(0, pl.hp-15);
        pl.invincible = 65;
        dmgFlashT = 12;
        exps.push({x:p.x, y:p.y, t:10, maxT:10, type:'fire'});
        if (pl.hp<=0) {
          pl.hp = 0;
          triggerGameOver();
        }
      }
    }
  }
  for (let i=projs.length-1; i>=0; i--) if (!projs[i].alive) projs.splice(i,1);
  for (let i=exps.length-1;  i>=0; i--) { exps[i].t--; if (exps[i].t<=0) exps.splice(i,1); }

  // Damage flash effect
  if (dmgFlashT > 0) {
    dmgFlashT--;
    dmgFlash.className = 'hit';
  } else {
    dmgFlash.className = '';
  }
}


const RESPAWN_POS=[[13,13],[13,2],[7,1],[1,7],[14,7],[7,14]];

function stepEnemy() {
  if (gameOverActive) return;
  if (en.dead) {
    if (--en.respT<=0) {
      const [rx,ry]=RESPAWN_POS[kills%RESPAWN_POS.length];
      const maxHp = en.maxHp;
      en = { x:rx+0.5, y:ry+0.5, a:Math.PI, hp:maxHp, maxHp, fcd:90, dead:false, spr_seq:0, spr_seq_time:0, respT:0, flashT:0,
             walkFrame:0, walkFrameTime:0, atkFrame:0, atkFrameTime:0, isAttacking:false, walkDir:0, gibDeath:false };
      enemies.push(en);
    }
    return;
  }
  if (en.flashT>0) en.flashT--;

  const inp=getInput(), ideal=getIdeal();
  let lsum=0;
  for (let i=0; i<TRAINING_STEPS_PER_FRAME; i++) { lsum+=nnTrain(inp,ideal); steps++; }
  curLoss=lsum/TRAINING_STEPS_PER_FRAME;
  lossHist.push(curLoss);
  if (lossHist.length>220) lossHist.shift();

  const out=nnForward(inp);
  lastOut=new Float32Array(out);

  const ETURN=0.065, ESPD=0.022;
  if (out[0]>0.5) en.a-=ETURN;
  if (out[1]>0.5) en.a+=ETURN;

  const ca=Math.cos(en.a), sa=Math.sin(en.a);
  const sca=Math.cos(en.a+Math.PI/2), ssa=Math.sin(en.a+Math.PI/2);
  let mvx=0, mvy=0;
  if (out[2]>0.5) { mvx+=ca*ESPD;      mvy+=sa*ESPD; }
  if (out[3]>0.5) { mvx-=ca*ESPD*0.65; mvy-=sa*ESPD*0.65; }
  if (out[4]>0.5) { mvx-=sca*ESPD*0.8; mvy-=ssa*ESPD*0.8; }
  if (out[5]>0.5) { mvx+=sca*ESPD*0.8; mvy+=ssa*ESPD*0.8; }

  const px=en.x, py=en.y;
  const nx=en.x+mvx, ny=en.y+mvy;
  if (!collidesAt(nx,en.y)) en.x=nx;
  if (!collidesAt(en.x,ny)) en.y=ny;

  // Drive walk animation from actual displacement projected onto facing direction
  const adx=en.x-px, ady=en.y-py;
  const fwd = adx*Math.cos(en.a) + ady*Math.sin(en.a);
  en.walkDir = Math.abs(adx)+Math.abs(ady) < 1e-5 ? 0 : fwd < 0 ? -1 : 1;

  // Advance attack animation; spawn fireball at last frame (h) using en.a
  if (en.isAttacking) {
    const now = performance.now();
    if (now - en.atkFrameTime >= ATK_FRAME_MS) {
      en.atkFrameTime = now;
      en.atkFrame++;
      if (en.atkFrame === ATK_FRAMES.length - 1) spawnFireball();
      if (en.atkFrame >= ATK_FRAMES.length) { en.isAttacking=false; en.atkFrame=0; }
    }
  } else if (out[6]>0.5 && en.fcd<=0) {
    en.fcd=80; en.isAttacking=true; en.atkFrame=0; en.atkFrameTime=performance.now();
  }
  if (en.fcd>0) en.fcd--;
}

function spawnFireball() {
  const a = en.a + (Math.random()-0.5)*0.15;
  projs.push({
    x: en.x+Math.cos(a)*0.55, y: en.y+Math.sin(a)*0.55,
    a, spd:0.10, type:'fireball', alive:true, life:0, maxLife:170
  });
}


// ══════════════════════════════════════════════
// MAIN LOOP
// ══════════════════════════════════════════════
function frame() {
  updatePlayer();
  updateProjectiles();
  stepEnemy();

  drawScene();
  drawEnemy();
  drawProjectiles();
  drawExplosions();
  drawMinimap();

  // Player invincibility flicker overlay
  if (pl.invincible>0 && (pl.invincible%8)<4) {
    drawRedFlicker();
  }

  drawLoss();
  updateHUD();
  requestAnimationFrame(frame);
}
