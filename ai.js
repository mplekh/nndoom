'use strict';

// ══════════════════════════════════════════════
// NN ARCHITECTURE — change these to reconfigure
// ══════════════════════════════════════════════
const INPUT_SIZE = 10;   // inputs
const HIDDEN_LAYER_1_SIZE = 64;
const HIDDEN_LAYER_2_SIZE = 16;
const OUTPUT_SIZE = 7;
const LEARNING_RATE = 0.012;
const TRAINING_STEPS_PER_FRAME = 1;
const OUTPUT_LABELS = ['ROT←','ROT→','FWD','BACK','STR←','STR→','FIRE'];
const INPUT_LABELS  = ['dx','dy','dist','sinA','cosA','LOS','rx','ry','rd','ehp'];

// ══════════════════════════════════════════════
// NEURAL NETWORK — 3-layer (INPUT → HIDDEN1 → HIDDEN2 → OUTPUT)
// ══════════════════════════════════════════════
const createRandomFloat32Array = (length, scale) => {
  const buffer = new Float32Array(length);
  for (let i = 0; i < length; i++) buffer[i] = (Math.random() * 2 - 1) * scale;
  return buffer;
};
const sig = x => 1/(1+Math.exp(-Math.max(-15,Math.min(15,x))));

const nn = {
  weightsInputToHidden1:createRandomFloat32Array(HIDDEN_LAYER_1_SIZE*INPUT_SIZE,0.3), biasesHidden1:new Float32Array(HIDDEN_LAYER_1_SIZE),
  weightsHidden1ToHidden2:createRandomFloat32Array(HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE,0.3), biasesHidden2:new Float32Array(HIDDEN_LAYER_2_SIZE),
  weightsHidden2ToOutput:createRandomFloat32Array(OUTPUT_SIZE*HIDDEN_LAYER_2_SIZE,0.25), biasesOutput:new Float32Array(OUTPUT_SIZE),
  momentW1:new Float32Array(HIDDEN_LAYER_1_SIZE*INPUT_SIZE),  momentVW1:new Float32Array(HIDDEN_LAYER_1_SIZE*INPUT_SIZE),
  momentB1:new Float32Array(HIDDEN_LAYER_1_SIZE),               momentVB1:new Float32Array(HIDDEN_LAYER_1_SIZE),
  momentW2:new Float32Array(HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE), momentVW2:new Float32Array(HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE),
  momentB2:new Float32Array(HIDDEN_LAYER_2_SIZE),               momentVB2:new Float32Array(HIDDEN_LAYER_2_SIZE),
  momentW3:new Float32Array(OUTPUT_SIZE*HIDDEN_LAYER_2_SIZE),  momentVW3:new Float32Array(OUTPUT_SIZE*HIDDEN_LAYER_2_SIZE),
  momentB3:new Float32Array(OUTPUT_SIZE),                        momentVB3:new Float32Array(OUTPUT_SIZE),
  linearHidden1:new Float32Array(HIDDEN_LAYER_1_SIZE), activationHidden1:new Float32Array(HIDDEN_LAYER_1_SIZE),
  linearHidden2:new Float32Array(HIDDEN_LAYER_2_SIZE), activationHidden2:new Float32Array(HIDDEN_LAYER_2_SIZE),
  outputActivations:new Float32Array(OUTPUT_SIZE),
  step:1,
};

function nnForward(inp) {
  for (let h=0; h<HIDDEN_LAYER_1_SIZE; h++) {
    let sum = nn.biasesHidden1[h];
    for (let i=0; i<INPUT_SIZE; i++) sum += nn.weightsInputToHidden1[h*INPUT_SIZE + i] * inp[i];
    nn.linearHidden1[h] = sum;
    nn.activationHidden1[h] = sum > 0 ? sum : 0;
  }
  for (let h=0; h<HIDDEN_LAYER_2_SIZE; h++) {
    let sum = nn.biasesHidden2[h];
    for (let i=0; i<HIDDEN_LAYER_1_SIZE; i++) sum += nn.weightsHidden1ToHidden2[h*HIDDEN_LAYER_1_SIZE + i] * nn.activationHidden1[i];
    nn.linearHidden2[h] = sum;
    nn.activationHidden2[h] = sum > 0 ? sum : 0;
  }
  for (let o=0; o<OUTPUT_SIZE; o++) {
    let sum = nn.biasesOutput[o];
    for (let h=0; h<HIDDEN_LAYER_2_SIZE; h++) sum += nn.weightsHidden2ToOutput[o*HIDDEN_LAYER_2_SIZE + h] * nn.activationHidden2[h];
    nn.outputActivations[o] = sig(sum);
  }
  return nn.outputActivations;
}

const gradW1 = new Float32Array(HIDDEN_LAYER_1_SIZE*INPUT_SIZE), gradB1 = new Float32Array(HIDDEN_LAYER_1_SIZE);
const gradW2 = new Float32Array(HIDDEN_LAYER_2_SIZE*HIDDEN_LAYER_1_SIZE), gradB2 = new Float32Array(HIDDEN_LAYER_2_SIZE);
const gradW3 = new Float32Array(OUTPUT_SIZE*HIDDEN_LAYER_2_SIZE), gradB3 = new Float32Array(OUTPUT_SIZE);

function nnBackward(inp, tgt) {
  let loss = 0;
  const deltaOutput = new Float32Array(OUTPUT_SIZE);
  for (let o=0; o<OUTPUT_SIZE; o++) {
    const error = nn.outputActivations[o] - tgt[o];
    loss += error * error;
    deltaOutput[o] = 2 * error * nn.outputActivations[o] * (1 - nn.outputActivations[o]);
  }
  const deltaHidden2 = new Float32Array(HIDDEN_LAYER_2_SIZE);
  for (let o=0; o<OUTPUT_SIZE; o++) {
    gradB3[o] += deltaOutput[o];
    for (let h=0; h<HIDDEN_LAYER_2_SIZE; h++) {
      gradW3[o*HIDDEN_LAYER_2_SIZE + h] += deltaOutput[o] * nn.activationHidden2[h];
      deltaHidden2[h] += deltaOutput[o] * nn.weightsHidden2ToOutput[o*HIDDEN_LAYER_2_SIZE + h];
    }
  }
  const deltaHidden1 = new Float32Array(HIDDEN_LAYER_1_SIZE);
  for (let h=0; h<HIDDEN_LAYER_2_SIZE; h++) {
    const dz = nn.linearHidden2[h] > 0 ? deltaHidden2[h] : 0;
    gradB2[h] += dz;
    for (let i=0; i<HIDDEN_LAYER_1_SIZE; i++) {
      gradW2[h*HIDDEN_LAYER_1_SIZE + i] += dz * nn.activationHidden1[i];
      deltaHidden1[i] += dz * nn.weightsHidden1ToHidden2[h*HIDDEN_LAYER_1_SIZE + i];
    }
  }
  for (let h=0; h<HIDDEN_LAYER_1_SIZE; h++) {
    const dz = nn.linearHidden1[h] > 0 ? deltaHidden1[h] : 0;
    gradB1[h] += dz;
    for (let i=0; i<INPUT_SIZE; i++) gradW1[h*INPUT_SIZE + i] += dz * inp[i];
  }
  return loss;
}

function adamStep(parameter, gradient, momentum, velocity) {
  const beta1 = 0.75, beta2 = 0.99, epsilon = 1e-8, timestep = nn.step;
  for (let i=0; i<parameter.length; i++) {
    momentum[i] = beta1 * momentum[i] + (1 - beta1) * gradient[i];
    velocity[i] = beta2 * velocity[i] + (1 - beta2) * gradient[i] * gradient[i];
    const correctedMomentum = momentum[i] / (1 - Math.pow(beta1, timestep));
    const correctedVelocity = velocity[i] / (1 - Math.pow(beta2, timestep));
    parameter[i] -= LEARNING_RATE * correctedMomentum / (Math.sqrt(correctedVelocity) + epsilon);
    gradient[i] = 0;
  }
}

function nnTrain(inp, tgt) {
  nnForward(inp);
  const loss = nnBackward(inp, tgt);
  nn.step++;
  adamStep(nn.weightsInputToHidden1, gradW1, nn.momentW1, nn.momentVW1);
  adamStep(nn.biasesHidden1, gradB1, nn.momentB1, nn.momentVB1);
  adamStep(nn.weightsHidden1ToHidden2, gradW2, nn.momentW2, nn.momentVW2);
  adamStep(nn.biasesHidden2, gradB2, nn.momentB2, nn.momentVB2);
  adamStep(nn.weightsHidden2ToOutput, gradW3, nn.momentW3, nn.momentVW3);
  adamStep(nn.biasesOutput, gradB3, nn.momentB3, nn.momentVB3);
  return loss;
}

// ══════════════════════════════════════════════
// ENEMY AI
// ══════════════════════════════════════════════
const MAX_D=18;
let steps=0, curLoss=1.0;
const lossHist=[];
let lastOut=new Float32Array(OUTPUT_SIZE);

function nearestRocket() {
  let best=null, bd=Infinity;
  for (const p of projs) {
    if (!p.alive||p.type!=='rocket') continue;
    const d=Math.hypot(p.x-en.x, p.y-en.y);
    if (d<bd) { bd=d; best=p; }
  }
  return best?{...best,dist:bd}:null;
}

function isApproaching(p) {
  return (en.x-p.x)*Math.cos(p.a)*p.spd + (en.y-p.y)*Math.sin(p.a)*p.spd > 0;
}

function dodgeDir(p) {
  return Math.cos(p.a)*(en.y-p.y) - Math.sin(p.a)*(en.x-p.x) > 0 ? 1 : -1;
}

function getInput() {
  const dx=(pl.x-en.x)/MAX_D, dy=(pl.y-en.y)/MAX_D;
  const dist=Math.hypot(dx,dy);
  const los=hasLOS(en.x,en.y,pl.x,pl.y)?1:0;
  const nr=nearestRocket();
  const rx=nr?(nr.x-en.x)/MAX_D:0;
  const ry=nr?(nr.y-en.y)/MAX_D:0;
  const rd=nr?Math.min(1,nr.dist/MAX_D):1;
  return [dx,dy,dist,Math.sin(en.a),Math.cos(en.a),los,rx,ry,rd,en.hp/en.maxHp];
}

function getIdeal() {
  const dx=pl.x-en.x, dy=pl.y-en.y;
  const dist=Math.hypot(dx,dy);
  const los=hasLOS(en.x,en.y,pl.x,pl.y);
  const targetOutputs=new Float32Array(OUTPUT_SIZE);
  const nr=nearestRocket();

  // Use pathfinding when no LOS to navigate around walls
  let navDx = dx, navDy = dy;
  if (!los) {
    const nav = pathDir(en.x, en.y, pl.x, pl.y);
    navDx = nav.dx; navDy = nav.dy;
  }

  let da=Math.atan2(navDy,navDx)-en.a;
  while(da> Math.PI) da-=2*Math.PI;
  while(da<-Math.PI) da+=2*Math.PI;

  if (nr && nr.dist<10 && isApproaching(nr)) {
    // Dodge incoming rocket
    const dd=dodgeDir(nr);
    targetOutputs[dd>0?5:4]=1.0;  // strafe right or left
    targetOutputs[3]=0.6;         // back away
    if (da> 0.15) targetOutputs[1]=0.45;
    else if (da<-0.15) targetOutputs[0]=0.45;
  } else {
    // Hunt player
    if      (da> 0.1) targetOutputs[1]=1;
    else if (da<-0.1) targetOutputs[0]=1;
    if      (dist>2.5) targetOutputs[2]=1;
    else if (dist<1.2) targetOutputs[3]=1;
    // Strafe while approaching for harder-to-hit movement
    if (dist>2.5 && dist<8) {
      // Alternate strafe direction based on steps
      targetOutputs[(steps>>5)&1 ? 4 : 5] = 0.5;
    }
    // Fire on LOS only
    if (los && dist<11 && Math.abs(da)<0.5) targetOutputs[6]=1;
  }
  return targetOutputs;
}
