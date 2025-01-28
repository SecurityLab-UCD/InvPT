from datasets import load_dataset
ds = load_dataset("code_search_net", "python", split='train')

import json

output_file_path = "./all_python.jsonl"
with open(output_file_path, "w") as f:
    for csn in ds:        
        # csn <class 'dict'>
        f.write(json.dumps(csn) + "\n")