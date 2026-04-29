# Reasoning Trajectory Viewer

This folder stores per-run reasoning trace JSONL files and a small static HTML viewer.

## Start the viewer

From the repo root:

```bash
bash reasoning-trajectory/start_viewer.sh 8765
```

The script serves `reasoning-trajectory/` as the HTTP root and prints the viewer URL:

```text
http://localhost:8765/viewer.html
```

## Remote access

If the browser is on a local machine and the server is remote, open an SSH tunnel from the local machine:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

Then open this locally:

```text
http://localhost:8765/viewer.html
```

## Load traces

The viewer auto-discovers run folders and task `.jsonl` files under this directory.

Use the `run folder` and `task file` dropdowns, then click `Load selected`.

If new trace files are created while the viewer is open, click `Refresh files`.

Manual path loading is still available as a fallback. Because the HTTP root is already `reasoning-trajectory/`, the shortest path is:

```text
<run-folder>/<task>.jsonl
```

For example:

```text
test_gpt-oss-20b-base-no-tool_20260428_201656.log/DILI.jsonl
```

