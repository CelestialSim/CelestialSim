---
name: godot-start-scene
description: Start a Godot scene safely from CLI, capture output to a log file, and extract benchmark/debug lines. Use for scene launch, perf checks, mesh generation timing, and reproducible log-based runs.
---

## Purpose

Use this skill when a task requires running a Godot scene and collecting deterministic output without relying on VS Code's interactive terminal rendering.

Keywords: Godot scene run, benchmark, performance, generation time, log file, CLI, stage_benchmark.

## What this skill provides

- A standard command flow to launch a scene through CLI.
- Automatic logging to `debug/logs/*.log`.
- Quick extraction of benchmark lines such as:
	- `Mesh Triangles: ...`
	- `Completed in ... ms`
	- `Benchmark orbit ...`

## Required script

This skill relies on:

- `debug/run_scene_with_log.sh`

The script:

1. Resolves Godot binary from `GODOT_PATH` or default local install.
2. Runs a chosen scene with `--path`.
3. Writes full output to a timestamped log file.
4. Prints benchmark-specific lines at the end.

## Usage

From project root:

```bash
./debug/run_scene_with_log.sh "$PWD" "res://scenes/stage_benchmark.tscn"
```

Optional custom log directory:

```bash
./debug/run_scene_with_log.sh "$PWD" "res://scenes/stage_benchmark.tscn" "$PWD/debug/logs"
```

## Agent checklist

1. Ensure target scene exists.
2. Ensure script is executable.
3. Run the script with explicit project path and scene path.
4. Read the newest `debug/logs/*.log` file.
5. Report:
	 - orbit start/end lines,
	 - mesh triangle count lines,
	 - generation timing (`Completed in ... ms`).

## Example output to report

- `Benchmark orbit started for 5.00s`
- `Mesh Triangles: 100000`
- `Completed in 47 ms`
- `Benchmark orbit completed in 5.001s`