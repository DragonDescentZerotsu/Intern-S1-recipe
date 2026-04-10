#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TREE_DIR = THIS_DIR.parent
if str(TREE_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_DIR))

from task_semantics import load_task_label_semantics


DEFAULT_TEMPLATE_PATH = THIS_DIR / "prompt_templates" / "rewrite_tree_draft_to_cot_prompt_template.md"
DEFAULT_REASONING_ROOT = TREE_DIR / "tree_reasoning_processes"
DEFAULT_PLAYBOOK_ROOT = TREE_DIR.parent.parent / "playbooks" / "tree_thresholds"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a filled prompt for rewriting one tree draft into a high-quality CoT."
    )
    parser.add_argument("--reasoning-json", default=None, help="Path to one reasoning JSON file")
    parser.add_argument("--experiment-name", default=None, help="Experiment name under tree_reasoning_processes")
    parser.add_argument("--task", default=None, help="Task name under the experiment directory")
    parser.add_argument("--feature-set", default=None, help="Feature-set directory name under the task directory")
    parser.add_argument("--sample-tag", default=None, help="Sample tag such as train_sample_0__top30__trees5")
    parser.add_argument("--tree-rank", type=int, default=1, help="1-based candidate tree rank to render")
    parser.add_argument(
        "--reasoning-root",
        default=str(DEFAULT_REASONING_ROOT),
        help="Root directory containing tree_reasoning_processes",
    )
    parser.add_argument(
        "--playbook-root",
        default=str(DEFAULT_PLAYBOOK_ROOT),
        help="Directory containing task playbooks, named like <task>.md",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to the rewrite prompt template",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where filled rewrite prompts will be written",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path for the filled prompt",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_reasoning_json_path(args: argparse.Namespace) -> Path:
    if args.reasoning_json:
        path = Path(args.reasoning_json).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        return path

    missing = [
        name
        for name in ("experiment_name", "task", "feature_set", "sample_tag")
        if getattr(args, name) is None
    ]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Either --reasoning-json or {joined} must be provided")

    reasoning_root = Path(args.reasoning_root).expanduser()
    if not reasoning_root.is_absolute():
        reasoning_root = (Path.cwd() / reasoning_root).resolve()
    else:
        reasoning_root = reasoning_root.resolve()

    return (
        reasoning_root
        / args.experiment_name
        / args.task
        / args.feature_set
        / f"{args.sample_tag}.json"
    )


def load_playbook_text(task: str, playbook_root: Path) -> str:
    playbook_path = playbook_root / f"{task}.md"
    if not playbook_path.exists():
        raise FileNotFoundError(f"Playbook not found for task {task}: {playbook_path}")
    return playbook_path.read_text(encoding="utf-8").strip()


def render_template(template_text: str, replacements: dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def resolve_output_path(
    *,
    args: argparse.Namespace,
    reasoning_payload: dict[str, object],
    reasoning_json_path: Path,
    tree_rank: int,
) -> Path:
    if args.output_path:
        path = Path(args.output_path).expanduser()
        if not path.is_absolute():
            return (Path.cwd() / path).resolve()
        return path.resolve()

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    else:
        output_root = output_root.resolve()

    experiment_name = reasoning_json_path.parents[2].name
    task = str(reasoning_payload.get("task") or "unknown_task")
    feature_set = str(reasoning_payload.get("feature_set_name") or reasoning_json_path.parent.name)
    sample_stem = reasoning_json_path.stem
    output_dir = output_root / experiment_name / task / feature_set / "rewrite_prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{sample_stem}__tree{tree_rank}_rewrite_prompt_filled.md"


def main() -> int:
    args = parse_args()
    reasoning_json_path = resolve_reasoning_json_path(args)
    if not reasoning_json_path.exists():
        raise FileNotFoundError(f"Reasoning JSON not found: {reasoning_json_path}")

    reasoning_payload = load_json(reasoning_json_path)
    reasoning_schema = reasoning_payload.get("reasoning_schema")
    if not isinstance(reasoning_schema, dict):
        raise ValueError(f"reasoning_schema missing from {reasoning_json_path}")

    task = str(reasoning_payload.get("task") or reasoning_schema.get("task") or args.task or "")
    if not task:
        raise ValueError("Could not infer task from reasoning JSON")

    candidate_trees = reasoning_schema.get("candidate_trees", [])
    if not isinstance(candidate_trees, list) or not candidate_trees:
        raise ValueError(f"No candidate trees found in {reasoning_json_path}")
    if args.tree_rank <= 0:
        raise ValueError("--tree-rank must be positive")

    target_tree = None
    for tree_row in candidate_trees:
        if int(tree_row["tree_rank"]) == args.tree_rank:
            target_tree = tree_row
            break
    if target_tree is None:
        raise ValueError(f"Tree rank {args.tree_rank} not found in {reasoning_json_path}")

    label_semantics = load_task_label_semantics(task)
    if label_semantics is None:
        class_a_text = "class 0"
        class_b_text = "class 1"
    else:
        class_a_text = label_semantics[0]["text"]
        class_b_text = label_semantics[1]["text"]

    playbook_root = Path(args.playbook_root).expanduser()
    if not playbook_root.is_absolute():
        playbook_root = (Path.cwd() / playbook_root).resolve()
    else:
        playbook_root = playbook_root.resolve()
    playbook_text = load_playbook_text(task, playbook_root)

    template_path = Path(args.template).expanduser()
    if not template_path.is_absolute():
        template_path = (Path.cwd() / template_path).resolve()
    else:
        template_path = template_path.resolve()
    template_text = template_path.read_text(encoding="utf-8")

    filled_prompt = render_template(
        template_text,
        {
            "TASK_NAME": task,
            "CLASS_A_TEXT": class_a_text,
            "CLASS_B_TEXT": class_b_text,
            "THRESHOLD_PLAYBOOK": playbook_text,
            "PATH_LEVEL_REASONING_NOTE": str(target_tree["path_level_reasoning_note"]).strip(),
        },
    )

    output_path = resolve_output_path(
        args=args,
        reasoning_payload=reasoning_payload,
        reasoning_json_path=reasoning_json_path,
        tree_rank=args.tree_rank,
    )
    output_path.write_text(filled_prompt, encoding="utf-8")

    print(
        json.dumps(
            {
                "reasoning_json": str(reasoning_json_path),
                "task": task,
                "tree_rank": args.tree_rank,
                "playbook_path": str(playbook_root / f"{task}.md"),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
