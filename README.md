# NEURAL DOOM

> **Content warning:** blood and gore (PG-13).

A browser game and AI demo:

- Doom-style 2.5D raycast arena
- Player rocket launcher vs enemy fireballs
- Enemy controlled by a small neural network trained live in-game with backpropagation
- HUD with health, training loss plot, learning phase progress, and active outputs

## Files

| File | Purpose |
|---|---|
| `neural-doom.html` | Page shell, styles, canvas, HUD markup |
| `map.js` | Map data and tile lookup |
| `ai.js` | Neural network, training loop, teacher policy |
| `graphics.js` | Raycaster, sprite renderer, game state, HUD logic |
| `gameloop.js` | Player input, projectiles, enemy step, main loop |
| `tro/` | Enemy sprite sheet (PNG frames) |

## Run

No build step is required.

1. Open `neural-doom.html` directly in a modern browser, or
2. Serve the folder locally:

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080/neural-doom.html`.

## Controls

- `WASD`: Move
- `Arrow Left` / `Arrow Right`: Turn
- `Arrow Up` / `Arrow Down`: Forward / backward (alternative, run when pressed simulaneously with W/S)
- `Space`: Fire rocket launcher

## How The AI Works

- Input features (`INPUT_SIZE = 10`): relative player position, distance, orientation, line-of-sight, nearest rocket info, enemy HP.
- Network: `INPUT_SIZE -> HIDDEN_LAYER_1_SIZE (ReLU) -> HIDDEN_LAYER_2_SIZE (ReLU) -> OUTPUT_SIZE (Sigmoid)`
- Outputs (`OUTPUT_SIZE = 7`): rotate left/right, move forward/back, strafe left/right, fire — the active ones light up on the HUD.
- Training: online supervised policy learning each frame (`TRAINING_STEPS_PER_FRAME` updates/frame), MSE loss, Adam optimizer.
- HUD shows current loss, phase progression, active outputs, and the neural architecture details.

## Tuning

The main constants are near the top of `ai.js`:

- `INPUT_SIZE`, `HIDDEN_LAYER_1_SIZE`, `HIDDEN_LAYER_2_SIZE`, `OUTPUT_SIZE`
- `LEARNING_RATE`
- `TRAINING_STEPS_PER_FRAME`
- `OUTPUT_LABELS`, `INPUT_LABELS`

Gameplay behaviour can also be tuned in `gameloop.js`:

- `PHASES` labels and thresholds
- Enemy movement/fire cooldown values in `stepEnemy()`
- Projectile values in `spawnRocket()`, `spawnFireball()`, and `updateProjectiles()`

## Game over and restart

When the player HP drops to zero a `GAME OVER` banner appears. The `RESTART` button resets player, enemy, projectiles, and net training stats without reloading the page.

## Credits

Enemy sprites by **Nue** — [Nue's Trash Textures and Sprites](https://www.doomworld.com/forum/topic/154344-nues-trash-textures-and-sprites/?tab=comments#comment-2951428) (Doomworld forums).

## Notes

- Internet access is needed for Google Fonts declared in the page header.
