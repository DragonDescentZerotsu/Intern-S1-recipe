import json
import logging
from pathlib import Path

from config import PROJECT_ROOT, EvalConfig

logger = logging.getLogger(__name__)

def apply_playbook(text: str, task_name: str, playbook_dir: str | None) -> str:
    if playbook_dir is None:
        return text
    pb_path = Path(playbook_dir) / f"{task_name}.txt"
    if pb_path.exists():
        with open(pb_path, 'r', encoding='utf-8') as f:
            pb_text = f.read().strip()
        if pb_text:
            return pb_text + "\n\n" + text.replace("think step by step", "follow the instructions in the playbook, think carefully,").replace("Instructions: Answer the following question about drug properties.\nContext: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.", "")
    return text

def load_train_data(data_dir: Path, exclude: tuple, task: str | None = None, playbook_dir: str | None = None) -> list[dict]:
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "DataPrepare" / "TDC_train_prompts_label_scaffold"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data dir found at {data_dir}")
    samples = []
    for fp in sorted(data_dir.glob("*.jsonl")):
        if task is not None:
            if fp.stem != task: continue
        elif fp.stem in exclude: continue
        with open(fp) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                item["task_name"] = fp.stem
                if "text" in item:
                    item["text"] = apply_playbook(item["text"], fp.stem, playbook_dir)
                samples.append(item)
    logger.info(f"Loaded {len(samples)} samples across {len(set(s['task_name'] for s in samples))} tasks")
    return samples

def load_test_data(ecfg: EvalConfig) -> dict[str, list[dict]]:
    if not ecfg.test_data_dir.exists():
        raise FileNotFoundError(f"Test dir {ecfg.test_data_dir} not found")
    tasks = {}
    playbook_dir = getattr(ecfg, "playbook_dir", None)
    for fp in sorted(ecfg.test_data_dir.glob("*.jsonl")):
        if fp.stem in ecfg.skip_tasks: continue
        if ecfg.eval_tasks and fp.stem not in ecfg.eval_tasks: continue
        with open(fp) as f:
            samps = []
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    if "text" in item:
                        item["text"] = apply_playbook(item["text"], fp.stem, playbook_dir)
                    samps.append(item)
        if ecfg.max_samples_per_task: samps = samps[:ecfg.max_samples_per_task]
        if samps: tasks[fp.stem] = samps
    logger.info(f"Eval: {sum(len(v) for v in tasks.values())} samples across {len(tasks)} tasks")
    return tasks
