
import json
import re
import os
import sys
from tqdm import tqdm
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

# Add project root to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import describe_high_level_fg_fragments

def precalculate_fgs():
    input_path = current_dir / 'Skin_Reaction_test_vanilla.jsonl'
    output_path = current_dir / 'Skin_Reaction_test_fg_desc.jsonl'
    
    print(f"Reading from {input_path}")
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Dictionary to store SMILES -> Description
    # We use a dictionary to avoid re-calculating for the same SMILES if it appears multiple times,
    # though the user mentioned "just for each line is fine", caching by SMILES is the most robust way for the agent to look it up.
    smiles_to_desc = {}
    
    print(f"Processing {len(data)} items...")
    
    for item in tqdm(data):
        text = item['text']
        # Extract SMILES
        match = re.search(r'<SMILES>(.*?)</SMILES>', text)
        if match:
            smiles = match.group(1)
            if smiles not in smiles_to_desc:
                try:
                    desc = describe_high_level_fg_fragments(smiles)
                    smiles_to_desc[smiles] = desc
                except Exception as e:
                    print(f"Error processing SMILES {smiles}: {e}")
                    smiles_to_desc[smiles] = f"Error: {e}"
        else:
            print(f"No SMILES found in text: {text[:50]}...")

    print(f"Saving {len(smiles_to_desc)} unique descriptions to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # We'll save it as a JSON object: { "smiles1": "desc1", "smiles2": "desc2" }
        # This is easy for the agent to load.
        json.dump(smiles_to_desc, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    precalculate_fgs()
