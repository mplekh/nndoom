# NEURAL DOOM

A single-file browser game and AI demo in [`neural-doom.html`](./neural-doom.html):

- Doom-style 2.5D raycast arena
- Player rocket launcher vs enemy fireballs
- Enemy controlled by a small neural network trained live in-game with backpropagation
- HUD with health, training loss plot, learning phase progress, and active outputs

## Files

- `neural-doom.html`: Complete game, renderer, enemy AI, neural-network training logic, and UI in one file.

## Run

No build step is required.

1. Open `neural-doom.html` directly in a modern browser, or
2. Serve the folder locally and open it in a browser:

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080/neural-doom.html`.

## Controls

- `WASD`: Move
- `Arrow Left` / `Arrow Right`: Turn
- `Arrow Up` / `Arrow Down`: Forward / backward (alternative)
- `Space`: Fire rocket launcher

## How The AI Works

- Input features (`INPUT_SIZE = 10`): relative player position, distance, orientation, line-of-sight, nearest rocket info, enemy HP.
- Network: `INPUT_SIZE -> HIDDEN_LAYER_1_SIZE (ReLU) -> HIDDEN_LAYER_2_SIZE (ReLU) -> OUTPUT_SIZE (Sigmoid)`
- Outputs (`OUTPUT_SIZE = 7`): rotate left/right, move forward/back, strafe left/right, fire—the active ones light up on the HUD.
- Training: online supervised policy learning each frame (`TRAINING_STEPS_PER_FRAME` updates/frame), MSE loss, Adam optimizer.
- HUD shows current loss, phase progression, active outputs, and the neural architecture details.

## Tuning

The main tuning constants are near the top of the `<script>` block:

- `INPUT_SIZE`, `HIDDEN_LAYER_1_SIZE`, `HIDDEN_LAYER_2_SIZE`, `OUTPUT_SIZE`
- `LEARNING_RATE`
- `TRAINING_STEPS_PER_FRAME`
- `OUTPUT_LABELS`, `INPUT_LABELS`

You can also tweak the phase thresholds, enemy movement/fire cooldowns (`stepEnemy()`), and projectile/constants (e.g., `spawnRocket()`, `spawnFireball()`, `updateProjectiles()`).

## Game over and restart

- When the player HP drops to zero the HUD flashes and a `GAME OVER` banner appears.
- A `RESTART` button layered inside the banner resets player/enemy/projectiles/net training stats so you can quickly try again without reloading.

Gameplay and behavior can also be tuned through:

- `PHASES` labels and thresholds
- Enemy movement/fire cooldown values in `stepEnemy()`
- Projectile values in `spawnRocket()`, `spawnFireball()`, and `updateProjectiles()`

## Notes

- This project is intentionally self-contained in one HTML file for easy experimentation.
- Internet access is needed for Google Fonts declared in the page header.
