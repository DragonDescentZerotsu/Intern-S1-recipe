import json
import logging
from pathlib import Path

from config import PROJECT_ROOT, EvalConfig

logger = logging.getLogger(__name__)

def load_train_data(data_dir: Path, exclude: tuple) -> list[dict]:
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "DataPrepare" / "TDC_train_prompts_label_scaffold"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data dir found at {data_dir}")
    samples = []
    for fp in sorted(data_dir.glob("*.jsonl")):
        if fp.stem in exclude: continue
        with open(fp) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                item["task_name"] = fp.stem
                samples.append(item)
    logger.info(f"Loaded {len(samples)} samples across {len(set(s['task_name'] for s in samples))} tasks")
    return samples

def load_test_data(ecfg: EvalConfig) -> dict[str, list[dict]]:
    if not ecfg.test_data_dir.exists():
        raise FileNotFoundError(f"Test dir {ecfg.test_data_dir} not found")
    tasks = {}
    for fp in sorted(ecfg.test_data_dir.glob("*.jsonl")):
        if fp.stem in ecfg.skip_tasks: continue
        if ecfg.eval_tasks and fp.stem not in ecfg.eval_tasks: continue
        with open(fp) as f:
            samps = [json.loads(l) for l in f if l.strip()]
        if ecfg.max_samples_per_task: samps = samps[:ecfg.max_samples_per_task]
        if samps: tasks[fp.stem] = samps
    logger.info(f"Eval: {sum(len(v) for v in tasks.values())} samples across {len(tasks)} tasks")
    return tasks
