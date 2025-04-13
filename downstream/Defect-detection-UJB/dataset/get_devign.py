from pathlib import Path
from tqdm import tqdm
import fire
import json
import os
import requests

DOWNLOAD_LINK = "https://drive.usercontent.google.com/download?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF&export=download&authuser=0"
DEVIGN_PATH = Path("./devign")

def download():
    response = requests.get(DOWNLOAD_LINK, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(DEVIGN_PATH / Path("original_dataset.json"), "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def preprocess():
    js_all=json.load(open(DEVIGN_PATH / Path("original_dataset.json")))

    train_index=set()
    valid_index=set()
    test_index=set()

    with open(DEVIGN_PATH / Path('train.txt')) as f:
        for line in f:
            line=line.strip()
            train_index.add(int(line))
                        
    with open(DEVIGN_PATH / Path('valid.txt')) as f:
        for line in f:
            line=line.strip()
            valid_index.add(int(line))
            
    with open(DEVIGN_PATH / Path('test.txt')) as f:
        for line in f:
            line=line.strip()
            test_index.add(int(line))
            
            
    with open(DEVIGN_PATH / Path('train.jsonl'),'w') as f:
        for idx,js in enumerate(js_all):
            if idx in train_index:
                js['idx']=idx
                f.write(json.dumps(js)+'\n')
                
    with open(DEVIGN_PATH / Path('valid.jsonl'),'w') as f:
        for idx,js in enumerate(js_all):
            if idx in valid_index:
                js['idx']=idx
                f.write(json.dumps(js)+'\n')
                
    with open(DEVIGN_PATH / Path('test.jsonl'),'w') as f:
        for idx,js in enumerate(js_all):
            if idx in test_index:
                js['idx']=idx
                f.write(json.dumps(js)+'\n')

def main():
    download()
    preprocess()

if __name__ == "__main__":
    fire.Fire(main)
