````skill
---
name: godot-screenshot-scene
description: Generate a wrapper scene that combines screenshot capture + target scene, run it through debug/run_scene_with_log.sh, and inspect output artifacts.
---

## Purpose

Use this skill when you need a deterministic screenshot of a scene after N seconds from startup and want the run logged in `debug/logs`.

Keywords: screenshot, viewport capture, benchmark snapshot, wrapper scene, Godot CLI.

## What this skill provides

- A reusable capture scene: `res://debug/scenes/ViewportScreenshotAfterDelay.tscn`
- Capture logic script: `res://debug/scripts/ViewportScreenshotAfterDelay.cs`
- Wrapper generator + runner: `debug/run_scene_with_screenshot.py`
- Standard execution path via `debug/run_scene_with_log.sh`

## Usage

From project root:

```bash
chmod +x ./debug/run_scene_with_screenshot.py
./debug/run_scene_with_screenshot.py "res://scenes/stage_benchmark.tscn" --delay 5
```

Optional arguments:

```bash
./debug/run_scene_with_screenshot.py \
  "res://scenes/stage_benchmark.tscn" \
  --delay 6.5 \
  --screenshot "res://debug/logs/screenshots/custom_benchmark.png" \
  --wrapper-scene "res://debug/scenes/generated/custom_wrapper.tscn"
```

## Agent checklist

1. Ensure target scene exists.
2. Build C# once after script changes: `dotnet build`.
3. Run `debug/run_scene_with_screenshot.py` with target scene + delay.
4. Read the newest `debug/logs/*.log` for runtime output.
5. Verify screenshot exists under `debug/logs/screenshots/`.

## Output to report

- Wrapper scene path generated.
- Screenshot path.
- Log file path.
- Any benchmark lines from run script output.
````
