
import time
from tools.AccFG import high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids
from accfg import AccFG

def test_speed():
    smiles = "CCOC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c1csc(NC(c2ccccc2)(c2ccccc2)c2ccccc2)n1"
    
    print("Testing original function (with re-init)...")
    start = time.time()
    for _ in range(5):
        high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles)
    end = time.time()
    print(f"5 calls took {end - start:.4f}s (Avg {(end - start)/5:.4f}s)")

    print("Testing reuse...")
    afg = AccFG(print_load_info=False)
    start = time.time()
    for _ in range(5):
        # We simulate the logic inside the function but reuse afg
        # We can't easily simulate it without copying code, but let's just measure AccFG init time.
        pass
    
    print("Testing AccFG init time...")
    start = time.time()
    for _ in range(5):
        _ = AccFG(print_load_info=False)
    end = time.time()
    print(f"5 inits took {end - start:.4f}s (Avg {(end - start)/5:.4f}s)")

if __name__ == "__main__":
    test_speed()
