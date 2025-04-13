from dataclasses import dataclass, asdict
from java_transform import augment_accumulatively
from multiprocessing import cpu_count
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import json
import fire
import logging
from pprint import pprint
logging.basicConfig(level=logging.INFO)

@dataclass
class DevignProgram:
    idx: str
    func: str
    target: int

def preprocess(jsonl_path: Path) -> list[DevignProgram]:
    programs: list[DevignProgram] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            decoded = json.loads(line)
            programs.append(DevignProgram(str(decoded["idx"]), str(decoded["func"]),
                                      int(decoded["target"])))
    return programs


def process(dataset: list[DevignProgram]) -> list[DevignProgram]:
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

def write(out_path: Path, programs: list[DevignProgram]):
    logging.info("Writing augmented dataset")
    with Pool(cpu_count()) as pool:
        jsonl = pool.map(lambda j: json.dumps(asdict(j)), programs)
    with open(out_path, "w") as f:
        for j in jsonl:
            f.write(j + "\n")

def main(in_path: Path, out_path: Path):
    programs = preprocess(in_path)
    augmented = process(programs)
    pprint(augmented[0])
    write(out_path, augmented)

if __name__ == "__main__":
    fire.Fire(main)
