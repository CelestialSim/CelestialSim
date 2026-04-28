````skill
---
name: godot-screenshot-scene
description: Run one of the checked-in screenshot wrapper scenes through debug/run_scene_with_log.sh, then inspect the saved screenshots and log output.
---

## Purpose

Use this skill when you need a deterministic screenshot of a scene after startup and want the run logged in `debug/logs`.

Keywords: screenshot, viewport capture, benchmark snapshot, wrapper scene, Godot CLI, generated capture scene.

## What this skill provides

- Checked-in wrapper scenes under `res://debug/scenes/generated/`
- Single-shot capture script: `res://debug/scripts/capture_screenshot_after_delay.gd`
- Multi-shot capture scripts: `res://debug/scripts/capture_closeup.gd`, `res://debug/scripts/capture_multi_angles.gd`
- Standard execution path: `debug/run_scene_with_log.sh`

## Usage

From project root:

```bash
./debug/run_scene_with_log.sh "$PWD" "res://debug/scenes/generated/celestial_terrain_screenshot.tscn" "$PWD/debug/logs" 8
```

Other checked-in wrappers:

```bash
./debug/run_scene_with_log.sh "$PWD" "res://debug/scenes/generated/celestial_terrain_demo_closeup_capture.tscn" "$PWD/debug/logs" 12
./debug/run_scene_with_log.sh "$PWD" "res://debug/scenes/generated/celestial_terrain_demo_multi_capture.tscn" "$PWD/debug/logs" 12
```

If no suitable wrapper exists yet, create one under `debug/scenes/generated/` that:

1. Instances the target scene as a child node.
2. Adds a plain `Node` with one of the capture scripts above.
3. Sets the script exports (`delay_seconds` / `output_path` for single-shot, or `output_dir`, `camera_path`, `planet_path`, `settle_seconds` for multi-shot variants).

Example single-shot wrapper pattern:

```tscn
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://scenes/celestial_terrain_demo.tscn" id="1_scene"]
[ext_resource type="Script" path="res://debug/scripts/capture_screenshot_after_delay.gd" id="2_capture"]

[node name="Root" type="Node"]

[node name="Target" parent="." instance=ExtResource("1_scene")]

[node name="ScreenshotCapture" type="Node" parent="."]
script = ExtResource("2_capture")
delay_seconds = 5.0
output_path = "res://debug/logs/screenshots/custom_capture.png"
```

## Agent checklist

1. Ensure the target scene exists and check `debug/scenes/generated/` for an existing wrapper before creating a new one.
2. Pick the right wrapper type:
   - `capture_screenshot_after_delay.gd` for a single viewport capture.
   - `capture_closeup.gd` or `capture_multi_angles.gd` for multiple saved angles.
3. Run the wrapper scene via `debug/run_scene_with_log.sh`.
4. Read the newest `debug/logs/*.log` for runtime output and `SCREENSHOT_SAVED` lines.
5. Verify the expected PNG file or output directory exists under `debug/logs/screenshots/`.

## Output to report

- Wrapper scene path used.
- Screenshot path or output directory.
- Log file path.
- Any `SCREENSHOT_SAVED`, benchmark, warning, or error lines from the run.

## Notes

- This repo currently uses GDScript capture helpers, not C# capture scripts.
- The old `debug/run_scene_with_screenshot.py` wrapper-generator flow is not present in this checkout.
- Keep timeout values long enough for the chosen wrapper to save all shots before `run_scene_with_log.sh` kills the process.
````
