# DBZ effects for Zoom

- **Instant Transmission:** two fingers on your forehead, and your body tears into
  vertical strips with white streaks and vanishes. Three seconds later you tear back in,
  wearing glowing flame hair. Teleport again and you come back without it.
- **Kamehameha:** both hands open, wrists together, palms apart. Hold it and a blue
  energy ball charges between your palms, then a beam fires.
- **Power-up aura, three levels:** clench both fists. Electricity crackles around you,
  then a yellow flame aura ignites and your hair switches to the transformation hair.
  Fists again for a wine-red aura, and again for azure blue, each with its own hair.
  Press `p` to make it dissipate.
- **Energy disk:** hold one hand out, palm up, fingers stretched. A swirling energy
  disk forms above your palm. Swish that hand and it flies off that way.

```
webcam → MediaPipe (hands + face + body mask) → effect → OBS Virtual Camera → Zoom
```

## One-time setup (Mac)

1. **Install OBS** from https://obsproject.com (version 30 or newer).
   Open it, click **Start Virtual Camera**, approve the system prompt, then click
   **Stop Virtual Camera** and quit OBS. This only installs the camera; OBS does not
   need to be running afterwards.
2. **Install dependencies.** In Terminal, from this folder:
   ```bash
   chmod +x setup.sh run.sh
   ./setup.sh
   ```
   If you don't have Python: `brew install python@3.12`.

## Running it

```bash
./run.sh
```

The first run downloads three small MediaPipe models into `models/`. macOS will
ask to let Terminal use the camera; allow it (System Settings → Privacy & Security
→ Camera) and run again if it fails the first time.

In Zoom, go to **Settings → Video → Camera** and choose **OBS Virtual Camera**.

### The gestures

**Instant Transmission:** index and middle fingers extended and together, ring and
pinky curled, fingertips on your forehead. Hold for about a quarter second.

**Kamehameha:** both hands open, wrists touching, palms apart like a clam shell.
Hold it: the ball charges for about a second, then fires.
- The beam goes the way your **fingers point**. Hands off to one side with
  fingers pointing sideways gives a sideways beam.
- Open your hands **straight at the camera** (one hand's fingers up, the other's
  down) for a blast toward the viewer.
- The beam keeps going while you hold the pose (1.5 to 4 seconds) and follows your
  hands. Drop your hands to end it.
- Pressing your palms together (like praying) won't trigger it.

**Power-up aura:** both hands in view, both clenched into fists. Hold for half a
second. Thin electricity crackles around you for two seconds while the aura builds,
then it ignites with a flash. The aura stays on (with a subtle outline and an
occasional spark) until you press `p`. Teleporting also ends the aura, and you
come back with no hair overlay, whichever form you were in.

| Level | How | Aura and lightning | Hair (in `overlays/`) |
|-------|-----|--------------------|-----------------------|
| 1 | Fists, with no aura on | Yellow, pale blue lightning | `hair_transformation.png` or `.gif` |
| 2 | Fists again, while at level 1 | Wine red, red lightning | `hair_transformation_2.png` or `.gif` |
| 3 | Fists again, while at level 2 | Azure blue, blue lightning | `hair_transformation_3.png` or `.gif` |

Open your hands between power-ups: holding one long fist won't skip levels. When
you level up, the crackles switch to the new color right away and the aura shifts
color as it builds. Pressing `p` at any level drops you all the way back to normal.

**Energy disk:** one hand open, fingers stretched, palm facing **up** (raised above
you, or held out flat like carrying a plate). Hold it still for a moment and the
disk spins up above your palm (about 1.3 seconds). Once it's formed it follows your
hand. **Swish** that hand quickly and the disk flies off in the direction you
swished. Lower your hand for about a second without swishing and it fizzles out.
Teleporting also cancels it.

### Keys (preview window must be focused)

| Key | Action |
|-----|--------|
| `t` | Test Instant Transmission |
| `k` | Test Kamehameha |
| `p` | Power up / dissipate the aura |
| `u` | Go up one aura level (tests the levels without fists) |
| `x` | Energy disk: form one (at your palm if visible); press again to throw it |
| `h` | Turn the hair overlay on/off without teleporting |
| `d` | Debug view: hand landmarks, forehead target (green when you're on it), body mask, an arrow showing where the Kamehameha would fire (blue = sideways beam, yellow = toward the camera), and the energy disk's palm-up arrows |
| `r` | Reset the learned background |
| `q` | Quit |

### Options

```bash
./run.sh --gone 5        # stay invisible for 5 seconds
./run.sh --camera 1      # use a different webcam
./run.sh --no-vcam       # preview window only, no virtual camera
```

## Tips for a clean vanish

- **Teach it your background.** When you disappear, the script shows what's behind
  you. It learns that from every part of the scene your body isn't covering. Right
  after starting, lean out of frame for a second or two so it sees the whole room.
  Any spot it has never seen gets filled in with a blurred guess.
- Keep the camera still. If it moves, press `r` and lean out again.
- Even lighting and a background that contrasts with you help the body mask.
- Don't stack Zoom's virtual background or blur on top; they fight with the effect.

## Hair overlay

Every teleport toggles the hair: you reappear with it, and the next teleport takes
it away (it tears out with you). It follows your head, including tilting.

The built-in look is an original glowing violet flame hair. To use your own look,
save a PNG with a transparent background as `overlays/hair.png` and restart. Draw
it upright and centered, with the hairline about 80% of the way down the image.
If it sits too high, low, big or small on your head, adjust `PNG_ANCHOR_Y`,
`PNG_HAIRLINE` and `PNG_WIDTH` at the top of `hair_overlay.py`. The flame colors,
`FLAME_COUNT` and `FLAME_HEIGHT` are there too.

### Transformation hair

While the aura is on, the hair for that level replaces yours (see the table above).
Each can be a transparent PNG or GIF, and animated GIFs play their animation.
If a level's hair isn't there yet, the closest lower level's hair is used.

They're placed the same way as `hair.png` (same tips apply). To adjust all of them,
change `TRANSFORM_PNG_WIDTH`, `TRANSFORM_PNG_ANCHOR_Y` and `TRANSFORM_PNG_HAIRLINE`
in `hair_overlay.py`. To adjust one level, edit its entry in `TRANSFORM_HAIR` there:
`width` (size), `hairline` (height), `anchor_y`, `offset_x` (sideways, as seen in
the preview) and `mirror` (flip left-right). Placement changes need a restart. Files are picked up as soon as they're in the folder, and
reloaded if you replace them, with no restart needed.

## Sound

The built-in sounds in `sounds/` are original sounds the script synthesizes on first
run. To use your own, put an **MP3 or WAV** in `sounds/custom/` with the same name as
the sound it replaces (`.m4a` and `.aiff` work too). For example, `teleport_out.mp3`
replaces `teleport_out.wav`. It's used the next time that sound plays, with no
restart needed. Delete your file to go back to the built-in one.

| Name | When it plays |
|------|---------------|
| `teleport_out` | Instant Transmission: vanishing |
| `teleport_in` | Instant Transmission: reappearing |
| `kame_charge` | Kamehameha: charging |
| `kame_fire` | Kamehameha: the beam (cut off when the beam ends) |
| `powerup`, `powerup_2`, `powerup_3` | Aura power-up for each level (ignition at about 2 seconds) |
| `aura_loop` | Hum while the aura is on. Loops with no gap; the last 25 ms blend into the start (`LOOP_CROSSFADE` in `audio.py`) |
| `aura_down` | Aura dissipating |
| `disk_form` | Energy disk forming |
| `disk_hum` | Energy disk hovering (loops seamlessly) |
| `disk_fire` | Energy disk thrown |

The sounds play through your Mac's speakers. For other people to hear them:

- **Easy:** speakers on, and in Zoom set **Settings → Audio → Audio profile** to
  *Original sound for musicians* (otherwise noise suppression may cut them).
  Headphones will block this route.
- **Clean:** install the free [BlackHole](https://existential.audio/blackhole/)
  loopback driver, create an *Aggregate Device* in Audio MIDI Setup combining your
  mic and BlackHole, and choose it as Zoom's microphone. Then create a
  *Multi-Output Device* (your headphones/speakers + BlackHole) and set it as your
  Mac's sound output, so you hear the sounds and Zoom picks them up too.

## Tuning

Energy disk constants are at the top of `disk.py`: `HOLD_SECONDS`, `FORM_SECONDS`,
`SWISH_SPEED` (raise it if it fires by accident, lower it if your swishes don't
register), `FLY_SPEED`, `DISK_SIZE`, `DISK_ASPECT` (flatness), `DISK_LIFT`,
`PALM_UP_MIN` / `PALM_UP_KEEP` (how far the palm must face up to start / to stay),
`POSE_SMOOTHING` (lower = steadier), `SWISH_DISTANCE`, and the colors. In the debug view (`d`), each hand gets an arrow
showing how much its palm faces up, with a circle when the fingers are stretched,
turning green when the pose is recognised. If the arrow points *down* when your palm
faces up, set `PALM_FLIP = -1`.

Aura constants are at the top of `power_up.py`: `POWERUP_SECONDS`, `AURA_REACH`,
`AURA_RISE`, `AURA_OPACITY`, `OUTLINE_STRENGTH`, `CRACKLES_WHILE_ACTIVE`, `SHAKE`,
`REARM_SECONDS`, and `LEVELS`, which holds each level's colors (outer flames, inner
flames, outline, lightning, flash) and opacity.

Kamehameha constants are at the top of `kamehameha.py`: `CHARGE_SECONDS`,
`BEAM_MIN`/`BEAM_MAX`, `BEAM_WIDTH`, `SHAKE` (set 0 to turn off screen shake),
`FRONTAL_SPREAD`, and the colors.

Instant Transmission constants are at the top of `instant_transmission.py`:

- `GONE_SECONDS`, `HOLD_SECONDS`, `COOLDOWN_SECONDS`
- `T_OUT_TEAR`, `T_IN_TEAR`: how long the tear takes going out and coming back
- `STREAK_COUNT`, `STREAK_COLOR`, `STREAK_GLOW`: the white vertical lines
- `STRIP_SHIFT`, `SMEAR`: how far the body strips slide and smear vertically

If the gesture triggers too easily or not at all, open the debug view (`d`) and
adjust the `0.95` (distance to forehead) in `GestureTrigger.update` or the finger
thresholds in `is_two_finger_pose`.
