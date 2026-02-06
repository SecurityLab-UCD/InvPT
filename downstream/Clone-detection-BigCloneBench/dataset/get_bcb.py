import typer
import requests
import json
import logging
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from dataclasses import dataclass, asdict
from java_transform import augment_accumulatively

logging.basicConfig(level=logging.DEBUG)

RESOURCE_URL = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/refs/heads/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl"
ORIGINAL_PATH = "./original_data.jsonl"
AUG_PATH = "./augmented_data.jsonl"
NPROC = cpu_count()


@dataclass
class BCBProgram:
    idx: str
    func: str


def preprocess(raw: bytes) -> list[BCBProgram]:
    # Preprocess
    programs: list[BCBProgram] = []
    for line in raw.decode().splitlines():
        decoded = json.loads(line)
        programs.append(BCBProgram(str(decoded["idx"]), str(decoded["func"])))
    return programs


def process(dataset: list[BCBProgram]) -> list[BCBProgram]:
    logging.info("Preparing data for transformation")
    id_map = {i: entry for i, entry in enumerate(dataset)}
    programs = [(i, entry.func) for i, entry in id_map.items()]

    logging.info("Applying SPAT for transformation")
    transformed = augment_accumulatively(programs)

    succeed = []
    for i, code in transformed:
        entry = id_map[i]
        entry.func = code
        succeed.append(entry)

    return succeed


def main():
    logging.info("Fetching original dataset")
    response = requests.get(RESOURCE_URL)
    assert response.status_code == 200, "Failed to download data.jsonl"
    raw = response.content
    with open(ORIGINAL_PATH, "wb") as f:
        f.write(raw)

    logging.info("Augmenting dataset")
    dataset = preprocess(raw)
    aug_dataset = process(dataset)

    logging.info("Writing augmented dataset")
    with Pool(NPROC) as pool:
        augmented_jsonl = pool.map(lambda j: json.dumps(asdict(j)), aug_dataset)
    with open(AUG_PATH, "w") as f:
        for aj in augmented_jsonl:
            f.write(aj + "\n")


if __name__ == "__main__":
    typer.run(main)
