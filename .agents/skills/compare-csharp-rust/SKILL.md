````skill
---
name: compare-csharp-rust
description: Compare C# and Rust implementation output by capturing screenshots of matching scenes, then diffing them pixel-by-pixel with Python. Use for visual regression, parity validation, and refactor correctness checks.
---

## Purpose

Use this skill to verify that the Rust implementation produces visually equivalent output to the C# implementation. It captures screenshots of matching scene pairs, compares them pixel-by-pixel, and reports differences.

Keywords: compare, C#, Rust, screenshot diff, pixel comparison, parity, regression, visual test.

## Scene pairs

The project has matching scene pairs for each implementation:

| C# scene                                    | Rust scene                                    |
|----------------------------------------------|-----------------------------------------------|
| `res://scenes/celestial_csharp_demo.tscn`    | `res://scenes/celestial_rust_demo.tscn`       |
| `res://scenes/celestial_csharp_aggressive.tscn` | `res://scenes/celestial_rust_aggressive.tscn` |

## Prerequisites

- Python 3 with Pillow (`pip install Pillow`).
- The `godot-screenshot-scene` skill's infrastructure (`debug/run_scene_with_screenshot.sh`).
- Both C# and Rust implementations built and ready.

## Steps

### 1. Build

```bash
dotnet build
```

### 2. Capture C# screenshot

```bash
./debug/run_scene_with_screenshot.sh \
  "res://scenes/celestial_csharp_aggressive.tscn" \
  --delay 5 \
  --screenshot "$PWD/debug/logs/screenshots/csharp_aggressive.png"
```

### 3. Capture Rust screenshot

```bash
./debug/run_scene_with_screenshot.sh \
  "res://scenes/celestial_rust_aggressive.tscn" \
  --delay 5 \
  --screenshot "$PWD/debug/logs/screenshots/rust_aggressive.png"
```

### 4. Compare with Python

```bash
python3 -c "
from PIL import Image
import sys

rust = Image.open('debug/logs/screenshots/rust_aggressive.png')
csharp = Image.open('debug/logs/screenshots/csharp_aggressive.png')

print(f'Rust size: {rust.size}')
print(f'C# size: {csharp.size}')

if rust.size != csharp.size:
    print('ERROR: Image sizes differ — cannot compare.')
    sys.exit(1)

r_pixels = list(rust.getdata())
c_pixels = list(csharp.getdata())
diffs = 0
max_diff = 0
for r, c in zip(r_pixels, c_pixels):
    d = sum(abs(a - b) for a, b in zip(r, c))
    if d > 0:
        diffs += 1
        max_diff = max(max_diff, d)
total = len(r_pixels)
pct = 100.0 * diffs / total
print(f'Different pixels: {diffs} / {total} ({pct:.2f}%)')
print(f'Max channel diff sum: {max_diff}')
if pct > 10:
    print('WARN: Large difference detected — implementations may diverge.')
"
```

Replace `aggressive` with `demo` for the demo scene pair.

## Agent checklist

1. Ensure both scene files exist for the chosen pair.
2. Build with `dotnet build`.
3. Capture the C# screenshot (use `godot-screenshot-scene` skill).
4. Capture the Rust screenshot.
5. Check both logs for crashes or errors (`handle_crash`, `signal 11`, etc.).
6. Run the Python pixel comparison.
7. Report results: image sizes, different pixel count/percentage, max channel diff.
8. Flag if either run crashed (the screenshot may still be usable if captured before the crash).

## Output to report

- C# benchmark lines (e.g. `Mesh Triangles:`, `Completed in ... ms`).
- Rust benchmark lines.
- Whether either run crashed.
- Image sizes.
- Different pixels: count, percentage.
- Max channel diff sum.
- Pass/fail assessment (< 5% diff is generally acceptable for floating-point differences).
````
