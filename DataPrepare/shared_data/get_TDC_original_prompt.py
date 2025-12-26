'''
从 Skin_Reaction_test_FG.jsonl 中删除 Functional Groups: ... 部分，得到原始的 TDC prompt。
'''

import json
import re
import os
from pathlib import Path

current_dir = Path(__file__).parent

def remove_functional_groups():
    input_path = current_dir / 'Skin_Reaction_test_FG.jsonl'
    # Use the same directory for output but different filename
    output_path = current_dir / 'Skin_Reaction_test_vanilla.jsonl'
    
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in f_in:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Regex to remove the line starting with "Functional Groups:"
                # Matches a newline, then "Functional Groups:", then anything until the next newline
                # We use (?=\n) to peek at the next newline but not consume it, 
                # or we can consume it if we want to remove the newline too.
                # In the usage ...\nFunctional Groups: ...\nAnswer: ...
                # We want to remove "\nFunctional Groups: ..." so that it becomes ...\nAnswer: ...
                # So we match `\nFunctional Groups:.*` excluding the trailing newline of that line context if possible?
                # Actually, simply removing `\nFunctional Groups: [^\n]+` usually works.
                
                # Regex to wrap SMILES string in <SMILES> tags
                # Replaces Drug SMILES: '...' with Drug SMILES: <SMILES>...</SMILES>
                new_text = re.sub(r"Drug SMILES: '([^']+)'", r"Drug SMILES: <SMILES>\1</SMILES>", text)

                new_text = re.sub(r'\nFunctional Groups:.*', '', new_text)

                new_text = new_text.replace("Answer: (", "Please think step by step and use tools when necessary (**Don\'t use the same tool more than once**). Then put your final choice ((A) or (B)) after \"Answer:\"")
                
                data['text'] = new_text
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid json line")

    print(f"Finished processing {count} lines.")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    remove_functional_groups()
