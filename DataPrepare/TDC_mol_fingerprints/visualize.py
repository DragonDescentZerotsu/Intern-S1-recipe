import sys
import pickle
from pathlib import Path

def print_help():
    print("Usage: python visualize.py <path_to_pkl_file> [num_samples]")
    print("Example: python visualize.py Feature_Morgan_similarity/by_task/BBB_Martins/valid_similarity.pkl 2")

def visualize_pkl(file_path, num_samples=2):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return
        
    print(f"Loading {path.name}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Total keys (SMILES) in dictionary: {len(data)}")
    print("=" * 60)
    
    # Check if the dictionary is empty
    if not data:
        print("Dictionary is empty.")
        return
        
    # Get a few sample keys
    sample_keys = list(data.keys())[:num_samples]
    
    for i, key in enumerate(sample_keys):
        value = data[key]
        print(f"\n[{i+1}/{min(num_samples, len(data))}] SMILES Query: {key}")
        
        # Determine the type of the value
        if isinstance(value, dict):
            print("  Structure: Nested Dictionary (e.g. Similarity File)")
            for sub_key, sub_val in value.items():
                print(f"  - Key: '{sub_key}'")
                if isinstance(sub_val, list):
                    print(f"    - Type: List (Length: {len(sub_val)})")
                    if len(sub_val) > 0:
                        print(f"    - Top 3 items: {sub_val[:3]}")
                        if len(sub_val) > 3:
                            print("      ...")
                        # print(f"    - Bottom 1 item: {sub_val[-1:]}")
                else:
                    print(f"    - Value: {sub_val} (Type: {type(sub_val).__name__})")
        elif isinstance(value, list):
            print(f"  Structure: List (Length: {len(value)})")
            if len(value) > 0:
                print(f"  - Top 3 items: {value[:3]}")
                if len(value) > 3:
                     print("    ...")
        else:
            # Maybe it's an RDKit ExplicitBitVect (e.g., from train.pkl / valid.pkl)
            print(f"  Structure: {type(value).__name__}")
            if type(value).__name__ == "ExplicitBitVect":
                print(f"  - Number of ON bits: {value.GetNumOnBits()}")
                print(f"  - Total bits: {value.GetNumBits()}")
            else:
                print(f"  - Value: {value}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_help()
        sys.exit(0)
        
    pkl_file = sys.argv[1]
    n_samples = 2
    
    if len(sys.argv) > 2:
        try:
            n_samples = int(sys.argv[2])
        except ValueError:
            print(f"Warning: '{sys.argv[2]}' is not a valid number. Using default (2).")
            
    visualize_pkl(pkl_file, n_samples)
