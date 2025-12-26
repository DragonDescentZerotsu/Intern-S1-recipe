import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP

DATA_SOURCE_DIR = Path("../DataPrepare/TDC_train_prompts_label")

def main():
    files = list(DATA_SOURCE_DIR.glob("*.jsonl"))
    file_tasks = {f.stem for f in files if f.stem != "SAbDab_Chen"}
    
    map_tasks = set(TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.keys())
    
    print(f"Total Files (excluding SAbDab_Chen): {len(file_tasks)}")
    print(f"Total Map Keys: {len(map_tasks)}")
    
    missing_in_map = file_tasks - map_tasks
    missing_in_files = map_tasks - file_tasks
    
    print("\nTasks in Files but MISSING in Map:")
    for t in sorted(missing_in_map):
        print(f"  - {t}")
        
    print("\nTasks in Map but MISSING in Files:")
    for t in sorted(missing_in_files):
        print(f"  - {t}")

if __name__ == "__main__":
    main()
