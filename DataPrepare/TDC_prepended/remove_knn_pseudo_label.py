import json
import shutil
from pathlib import Path


SOURCE_DIR = Path("DataPrepare/TDC_prepended/KNN_3_per_label")
TARGET_DIR = Path("DataPrepare/TDC_prepended/KNN_3_per_label_no_pseudo_label")

PREFIXES_TO_REMOVE = [
    "The pseudo label from naive Morgan fingerprint KNN prediction is (A). ",
    "The pseudo label from naive Morgan fingerprint KNN prediction is (B). ",
]

OLD_INSTRUCTION = (
    "You should compare the molecules carefully and decide to agree with the pseudo label or not.\n\n"
    'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
)

NEW_INSTRUCTION = (
    'Please think step by step, compare the molecules carefully, and then put ONLY your final choice ((A) or (B)) after "Answer:"'
)


def remove_pseudo_label_suffix(text: str) -> str:
    for prefix in PREFIXES_TO_REMOVE:
        if prefix in text:
            text = text.replace(prefix, "")
    return text.replace(OLD_INSTRUCTION, NEW_INSTRUCTION)


def process_jsonl_file(src_file: Path, dst_file: Path) -> tuple[int, int]:
    total_count = 0
    modified_count = 0

    with src_file.open("r", encoding="utf-8") as fin, dst_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_count += 1
            record = json.loads(line)

            original_text = record.get("text", "")
            updated_text = remove_pseudo_label_suffix(original_text)
            if updated_text != original_text:
                modified_count += 1
                record["text"] = updated_text

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return total_count, modified_count


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    total_files = 0
    modified_files = 0
    total_records = 0
    modified_records = 0

    for src_path in SOURCE_DIR.rglob("*"):
        relative_path = src_path.relative_to(SOURCE_DIR)
        dst_path = TARGET_DIR / relative_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        should_process_jsonl = (
            src_path.suffix == ".jsonl"
            and src_path.parent.name in {"train", "valid"}
        )

        if should_process_jsonl:
            total_files += 1
            file_total_records, file_modified_records = process_jsonl_file(src_path, dst_path)
            total_records += file_total_records
            modified_records += file_modified_records
            if file_modified_records > 0:
                modified_files += 1
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Created: {TARGET_DIR}")
    print(f"Processed jsonl files: {total_files}")
    print(f"Modified jsonl files: {modified_files}")
    print(f"Processed records: {total_records}")
    print(f"Modified records: {modified_records}")


if __name__ == "__main__":
    main()
