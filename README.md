# NEURAL DOOM

> **Content warning:** blood and gore (PG-13).

A browser game and AI demo where **both sides are neural-network controlled**:

- Doom-style 2.5D raycast arena
- Two player weapons (Repeater, Gauss rifle) vs enemy fireballs
- **Enemy** controlled by a small MLP trained live with backpropagation
- **Player** controlled by a MicroGPT transformer with KV-cache attention, trained live via supervised learning from a heuristic teacher
- HUD with health, training loss plot, learning phase, active outputs, and GPT controller panel

## Dual Neural Network Architecture

| | **Enemy MLP** | **Player MicroGPT** |
|---|---|---|
| **Architecture** | 3-layer MLP (10→64→16→7) | 1-layer Transformer (208→32 embd, 4-head attn, MLP 128) |
| **Parameters** | ~1,800 | ~19,700 |
| **Input** | 10 game-state floats | 192 visual pixels (16×12) + 16 game-state floats |
| **Output** | 7 sigmoids (turn L/R, fwd/back, strafe L/R, fire) | 4 signed axes (move, turn, strafe, fire) |
| **Activation** | ReLU + sigmoid | ReLU, RMSNorm, softmax attention |
| **Training** | Supervised MSE vs heuristic, Adam | Supervised MSE vs heuristic, Adam |
| **Temporal context** | None (stateless) | KV-cache attention over last 16 frames |
| **Backprop** | Hand-coded per-layer | Tape autograd (reverse-mode AD) |

Both networks learn from a **heuristic teacher** each frame via MSE loss. The teachers share a BFS pathfinder (`pathDir` in `map.js`) that computes grid-shortest paths around walls, so both agents navigate obstacles instead of walking into them. The enemy MLP converges in seconds; the player transformer takes longer but learns to use visual input and temporal context.

## Files

| File | Purpose |
|---|---|
| `neural-doom.html` | Page shell, styles, canvas, HUD markup |
| `map.js` | Map data, tile lookup, BFS pathfinder |
| `ai.js` | Enemy neural network, training loop, teacher policy |
| `graphics.js` | Raycaster, sprite renderer, game state, HUD logic |
| `gameloop.js` | Player input, projectiles, enemy step, main loop |
| `microgpt-tape.js` | Tape autograd engine, transformer components, Adam optimizer |
| `microgpt-ctrl.js` | GPT controller: visual encoding, game state, KV cache, training, HUD |
| `tro/` | Enemy sprite sheet (PNG frames) |
| `wea/` | First-person weapon sprites (PNG frames) |

### MicroGPT Tape Engine (`microgpt-tape.js`)

A from-scratch autograd engine ported from `tapegpt.py`:

- **Tape class**: pre-allocated typed arrays (Float64Array/Int32Array, capacity 80K nodes), tracks values, gradients, children, and local derivatives
- **Primitives**: add, mul, mul_const, pow_const, div, log, relu, sigmoid, tanh
- **Transformer components**: `t_linear`, `t_rmsnorm`, `t_softmax`
- **GPT forward pass**: input projection → positional embedding → multi-head attention with detached KV cache → MLP → dual output heads (policy + value)
- **Adam optimizer**: β1=0.85, β2=0.99, bias-corrected

### GPT Controller (`microgpt-ctrl.js`)

- **Visual encoding**: 800×600 RGBA → 16×12 grayscale (5×5 sampling per block, BT.709 luma)
- **Game state**: 16 features (player HP, angle, enemy relative position/distance/angle, LOS, weapon cooldown, nearest fireball, previous action)
- **KV cache**: circular buffer of 16 past frames, detached (no gradient flow through past frames)
- **Training**: MSE loss against heuristic teacher, per-frame Adam update
- **4-axis output**: move (±forward/back), turn (±right/left), strafe (±right/left), fire (±shoot/hold)

## Run

No build step required. A local server is needed for pixel capture (cross-origin restriction on `file://`):

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080/neural-doom.html`.

## Controls

- `WASD`: Move
- `Arrow Left` / `Arrow Right`: Turn
- `Arrow Up` / `Arrow Down`: Forward / backward (alternative)
- `Space`: Fire
- `1`: Repeater
- `2`: Gauss rifle
- `3`: Cycle GPT mode (OBSERVE → ASSISTIVE → AUTONOMOUS)
- `4`: Toggle action visualization

## GPT Controller Modes

- **OBSERVE**: AI watches, player has full control
- **ASSISTIVE**: AI actions merge with player input (default)
- **AUTONOMOUS**: AI has full control, player can only switch weapons

## Heuristic Teacher

Both AIs learn from a shared teacher design that computes ideal actions from game state each frame:

- **Turn**: proportional to signed angle error toward target
- **Move**: forward when roughly facing target, back up if too close
- **Strafe**: dodge perpendicular to incoming projectiles (rockets for enemy, fireballs for player)
- **Fire**: only when aimed (angle error < threshold) AND line-of-sight is clear
- **Pathfinding**: when LOS is blocked, BFS on the 32×32 grid (`pathDir` in `map.js`) finds the shortest walkable path around walls; the teacher turns toward the next waypoint instead of walking into the obstacle

## How The Enemy AI Works

- Input features (`INPUT_SIZE = 10`): relative player position, distance, orientation, line-of-sight, nearest rocket info, enemy HP.
- Network: `10 → 64 (ReLU) → 16 (ReLU) → 7 (Sigmoid)`
- Outputs: rotate left/right, move forward/back, strafe left/right, fire — the active ones light up on the HUD.
- Training: online supervised learning each frame, MSE loss, Adam optimizer.

## How The Player GPT Works

- Input: 192 downsampled grayscale pixels + 16 game state features = 208-dim vector
- Projected to 32-dim embedding, positional encoding added, passed through 1-layer transformer (4 heads, head_dim=8, MLP with 128 hidden units)
- Attends to KV cache of up to 16 past frames for temporal context
- Outputs 4 continuous axes mapped to keys via dead-zone thresholds
- Trained via MSE against the heuristic teacher; loss converges within seconds

## Tuning

Enemy AI constants are near the top of `ai.js`:
- `INPUT_SIZE`, `HIDDEN_LAYER_1_SIZE`, `HIDDEN_LAYER_2_SIZE`, `OUTPUT_SIZE`
- `LEARNING_RATE`, `TRAINING_STEPS_PER_FRAME`

Player GPT config is in `microgpt-tape.js`:
- `GPT.n_embd`, `GPT.n_head`, `GPT.n_layer`, `GPT.block_size`, `GPT.feat_dim`, `GPT.n_actions`

Gameplay tuning in `gameloop.js` and `graphics.js`:
- Enemy movement/fire cooldown in `stepEnemy()`
- Weapon constants (damage, cooldown, sprites) near top of `graphics.js`

## Game over and restart

When player HP drops to zero a `GAME OVER` banner appears. The `RESTART` button resets player, enemy, projectiles, and training stats. The GPT's KV cache is cleared on restart.

## Credits

Enemy sprites by **Nue** — [Nue's Trash Textures and Sprites](https://www.doomworld.com/forum/topic/154344-nues-trash-textures-and-sprites/?tab=comments#comment-2951428) (Doomworld forums).

Weapon sprites — *Doom 2016 Weapon Sprites v2.0* by **Neccronixis** — [ZDoom forums](https://forum.zdoom.org/viewtopic.php?t=51919).

## Notes

- Internet access is needed for Google Fonts declared in the page header.
- A local HTTP server is required (not `file://`) for the GPT controller's pixel capture to work.
