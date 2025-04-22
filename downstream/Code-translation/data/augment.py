from fire import Fire
from java_transform.utils import Program as JavaProgram
from java_transform import TRANSFORMATION_MAP as JAVA_AUGMAP
import logging
logging.basicConfig(level=logging.INFO)

def augment_java(in_path: str, out_path: str):
    programs: list[JavaProgram] = []
    logging.info(f"Loading {in_path}")
    with open(in_path, 'r') as f:
        for idx, line in enumerate(f):
            programs.append((idx, line))
    for aug_type in JAVA_AUGMAP.keys():
        logging.info(f"Applying {aug_type}")
        programs = JAVA_AUGMAP[aug_type](programs)
    with open(out_path, 'w') as f:
        f.writelines([program[1] for program in programs])

def main():
    augment_java("./test.java-cs.txt.java", "./aug_test.java-cs.txt.java")
    augment_java("./train.java-cs.txt.java", "./aug_train.java-cs.txt.java")
    augment_java("./valid.java-cs.txt.java", "./aug_valid.java-cs.txt.java")


if __name__ == "__main__":
    Fire(main)
