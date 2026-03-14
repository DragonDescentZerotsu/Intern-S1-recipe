import pickle
from pathlib import Path

def test_similarities():
    base_dir = Path(__file__).parent / "Feature_Morgan_similarity" / "by_task" / "BBB_Martins"
    
    with open(base_dir / "valid_similarity.pkl", "rb") as f:
        valid_sim = pickle.load(f)
        
    with open(base_dir / "train_similarity.pkl", "rb") as f:
        train_sim = pickle.load(f)
        
    print(f"Valid Similarity keys: {len(valid_sim)}")
    print(f"Train Similarity keys: {len(train_sim)}")
    
    # Check Valid
    k1 = list(valid_sim.keys())[0]
    sims1_dict = valid_sim[k1]
    
    print(f"\nExample Valid SMILES: {k1}")
    for label in ["label_0", "label_1"]:
        sims = sims1_dict[label]
        print(f"[{label}] Number of comparisons: {len(sims)}")
        print(f"[{label}] Top 3 similarities: {sims[:3]}")
        print(f"[{label}] Bottom 3 similarities: {sims[-3:]}")
        # Verify sorting
        assert all(sims[i][0] >= sims[i+1][0] for i in range(len(sims)-1)), f"Valid Similarity ({label}) not sorted correctly!"
    
    # Check Train
    k2 = list(train_sim.keys())[0]
    sims2_dict = train_sim[k2]
    
    print(f"\nExample Train SMILES: {k2}")
    for label in ["label_0", "label_1"]:
        sims = sims2_dict[label]
        print(f"[{label}] Number of comparisons: {len(sims)}")
        print(f"[{label}] Top 3 similarities: {sims[:3]}")
        print(f"[{label}] Bottom 3 similarities: {sims[-3:]}")
        # Verify sorting
        assert all(sims[i][0] >= sims[i+1][0] for i in range(len(sims)-1)), f"Train Similarity ({label}) not sorted correctly!"
        # Verify self exclusion
        assert k2 not in [x[1] for x in sims], f"Train similarity ({label}) did NOT exclude self!"
    
    print("\nAll checks passed successfully!")

if __name__ == "__main__":
    test_similarities()
