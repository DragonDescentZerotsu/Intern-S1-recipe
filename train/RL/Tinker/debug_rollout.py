import json
import argparse
import os
import sys

try:
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

def print_rollout(data):
    if RICH_AVAILABLE:
        header = (
            f"Batch: {data.get('batch')} | Task: {data.get('task')} | Rollout Idx: {data.get('rollout_idx')}\n"
            f"Reward: {data.get('reward')} | GT: {data.get('gt')}\n"
            f"Tokens: {data.get('n_tokens')} | Response Tokens: {data.get('n_response_tokens')} | Turns: {data.get('n_turns')}"
        )
        console.print(Panel(header, title=f"Rollout ID: {data.get('rollout_idx')}", border_style="cyan"))
        
        messages = data.get("messages", [])
        if messages:
            # Reorder keys to display "thinking" before "tool_calls"
            ordered_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "thinking" in msg and "tool_calls" in msg:
                    ordered_msg = {}
                    for k, v in msg.items():
                        if k == "tool_calls":
                            continue
                        if k == "thinking":
                            ordered_msg["thinking"] = msg["thinking"]
                            ordered_msg["tool_calls"] = msg["tool_calls"]
                            continue
                        ordered_msg[k] = v
                    ordered_messages.append(ordered_msg)
                else:
                    ordered_messages.append(msg)
                    
            msg_json_str = json.dumps(ordered_messages, indent=2, ensure_ascii=False)
            console.print(Panel("Messages (JSON Format)", border_style="green"))
            console.print(JSON(msg_json_str))
        else:
            console.print("[bold yellow]No 'messages' key found in this rollout.[/bold yellow]")
            if "response_text" in data:
                console.print(Panel(data["response_text"], title="Response Text", border_style="yellow"))
    else:
        print("="*80)
        print(f"Batch: {data.get('batch')} | Task: {data.get('task')} | Rollout Idx: {data.get('rollout_idx')}")
        print(f"Reward: {data.get('reward')} | GT: {data.get('gt')}")
        print(f"Tokens: {data.get('n_tokens')} | Response Tokens: {data.get('n_response_tokens')} | Turns: {data.get('n_turns')}")
        print("-"*80)
        
        messages = data.get("messages", [])
        if messages:
            print("Messages (JSON Format):")
            # Reorder keys to display "thinking" before "tool_calls"
            ordered_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "thinking" in msg and "tool_calls" in msg:
                    ordered_msg = {}
                    for k, v in msg.items():
                        if k == "tool_calls":
                            continue
                        if k == "thinking":
                            ordered_msg["thinking"] = msg["thinking"]
                            ordered_msg["tool_calls"] = msg["tool_calls"]
                            continue
                        ordered_msg[k] = v
                    ordered_messages.append(ordered_msg)
                else:
                    ordered_messages.append(msg)
            print(json.dumps(ordered_messages, indent=2, ensure_ascii=False))
        else:
            print("No 'messages' key found.")
            if "response_text" in data:
                print("Response Text:")
                print(data["response_text"])
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="View and format JSONL rollout files.")
    parser.add_argument("file", help="Path to the rollout JSONL file (e.g. rollouts/batch_000000.jsonl)")
    parser.add_argument("-i", "--idx", type=int, help="Specific rollout_idx to display")
    parser.add_argument("-r", "--row", type=int, help="Specific line number (1-indexed) in the jsonl file to display")
    parser.add_argument("-l", "--limit", type=int, default=1, help="Number of rollouts to display if --idx or --row is not specified (default: 1)")
    parser.add_argument("-a", "--all", action="store_true", help="Display all rollouts in the file")

    args = parser.parse_args()

    filepath = args.file
    if not os.path.exists(filepath):
        # Try to look inside the rollouts directory just in case
        alt_path = os.path.join(os.path.dirname(__file__), "rollouts", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            print(f"Error: File not found at '{args.file}' or '{alt_path}'", file=sys.stderr)
            sys.exit(1)

    printed_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for row_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {row_num}: {e}", file=sys.stderr)
                continue
            
            if args.row is not None:
                if row_num == args.row:
                    print_rollout(data)
                    return
                continue

            if args.idx is not None:
                if data.get("rollout_idx") == args.idx:
                    print_rollout(data)
                    return
                continue
            
            print_rollout(data)
            printed_count += 1
            if not args.all and printed_count >= args.limit:
                break
                
    if args.row is not None:
        print(f"Row {args.row} not found in {filepath}.", file=sys.stderr)
    elif args.idx is not None:
        print(f"Rollout with idx {args.idx} not found in {filepath}.", file=sys.stderr)

if __name__ == "__main__":
    main()
