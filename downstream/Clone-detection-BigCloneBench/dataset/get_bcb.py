import fire
import requests

RESOURCE_URL = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/refs/heads/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl"
ORIGINAL_PATH = "./original_data.jsonl"

def main():
    # Fetch data.jsonl
    response = requests.get(RESOURCE_URL)
    assert response.status_code == 200, "Failed to download data.jsonl"
    raw = response.content
    with open(ORIGINAL_PATH, "wb") as f:
        f.write(raw)

if __name__ == "__main__":
    fire.Fire(main)
