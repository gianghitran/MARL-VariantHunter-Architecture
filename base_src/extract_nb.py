import json
import sys

def extract_code(nb_path, out_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                f.write(f"# CELL {i}\n")
                f.write(source)
                f.write("\n\n")

extract_code('01-rlhf-ppo.ipynb', 'ppo_extracted.py')
extract_code('unicorn.ipynb', 'unicorn_extracted.py')
