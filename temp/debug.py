import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path


DEFAULT_JSONL_PATH = Path("DataPrepare/TDC_prepended/KNN/train/BBB_Martins.jsonl")


def terminal_width() -> int:
    return max(88, shutil.get_terminal_size(fallback=(120, 40)).columns)


def hr(char: str = "=", width: int | None = None) -> str:
    return char * (width or terminal_width())


def wrap_block(text: str, width: int, indent: str = "") -> str:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        if not raw_line.strip():
            lines.append("")
            continue
        wrapped = textwrap.fill(
            raw_line,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(wrapped)
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {idx} 行不是合法 JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"文件中没有可读取的 JSONL 内容: {path}")
    return rows


def preview_text(text: str, limit: int = 78) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def choose_index(rows: list[dict]) -> int:
    print(hr())
    print("可选数据行")
    print(hr("-"))
    for idx, row in enumerate(rows, start=1):
        drug = row.get("drug", "N/A")
        label = row.get("Y", "N/A")
        text_preview = preview_text(str(row.get("text", "")))
        print(f"[{idx}] drug={drug} | Y={label}")
        print(f"    {text_preview}")
    print(hr("-"))

    while True:
        user_input = input(f"请输入要查看的行号 (1-{len(rows)}): ").strip()
        if user_input.isdigit():
            selected = int(user_input)
            if 1 <= selected <= len(rows):
                return selected - 1
        print("输入无效，请重新输入一个存在的行号。")


def print_section(title: str, content: str, width: int) -> None:
    print()
    print(f"[{title}]")
    print(hr("-", width))
    print(wrap_block(content, width=width))


def format_json_value(value, width: int) -> str:
    if isinstance(value, str):
        return wrap_block(value, width=width)
    return wrap_block(
        json.dumps(value, ensure_ascii=False, indent=2),
        width=width,
    )


def display_row(row: dict, row_number: int, source_path: Path) -> None:
    width = terminal_width()
    title = f"BBB_Martins 第 {row_number} 行"

    print(hr("=", width))
    print(title.center(width))
    print(hr("=", width))
    print(f"来源文件 : {source_path}")
    print(f"drug     : {row.get('drug', 'N/A')}")
    print(f"Y        : {row.get('Y', 'N/A')}")
    text_value = str(row.get("text", ""))
    print(f"text长度 : {len(text_value)} 字符")

    if text_value:
        print_section("text", text_value, width)

    other_keys = [key for key in row.keys() if key not in {"text", "drug", "Y"}]
    if other_keys:
        print()
        print("[其他字段]")
        print(hr("-", width))
        for key in other_keys:
            print(f"{key}:")
            print(format_json_value(row[key], width=width - 2))
            print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 JSONL 文件中的指定行，并在终端中进行美观展示。"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help=f"JSONL 文件路径，默认是: {DEFAULT_JSONL_PATH}",
    )
    parser.add_argument(
        "-n",
        "--line",
        type=int,
        help="要查看的行号（从 1 开始）。不传则进入交互选择模式。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rows = load_jsonl(args.file)
        if args.line is not None:
            if not 1 <= args.line <= len(rows):
                raise ValueError(
                    f"行号超出范围: {args.line}，当前文件共有 {len(rows)} 行。"
                )
            selected_index = args.line - 1
        else:
            selected_index = choose_index(rows)

        display_row(rows[selected_index], selected_index + 1, args.file)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
