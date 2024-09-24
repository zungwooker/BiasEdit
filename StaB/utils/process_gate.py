import os
import glob
import json
import argparse
import itertools

def process_gate(args: argparse.ArgumentParser, file_name: str):
    class_name_path = os.path.join(args.root_path, 'benchmarks', args.dataset, 'class_name.json') # benchmarks/{args.dataset}/class_name.json
    if os.path.exists(class_name_path):
        with open(class_name_path, 'r') as file:
            try:
                class_name = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f"class_name.json does not exist.\nPath: {class_name_path}")
    
    if '.json' in file_name:
        for label, bias in itertools.product(class_name, ['align', 'conflict']):
            json_path = os.path.join(args.root_path, args.preproc, args.dataset, args.conflict_ratio+'pct', bias, label, 'jsons', file_name)
            if not os.path.exists(json_path):
                print(f"[Warning] Process gate: File missing {json_path}")
                print(f"[Work] Process gate: Working on {file_name}...")
                return False
    elif file_name == 'png':
        print("[WIP] Process gate: wip for edit...")
        return False
        
    print(f"[Done] Process gate: All files exists. {file_name}")
    return True
            