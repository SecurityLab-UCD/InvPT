import json
import fire

JSON_ENCODING = "latin-1"

def main(test_file):
    with open(test_file, "r", encoding=JSON_ENCODING) as f:
        all_jsons = [json.loads(json_line) for json_line in f]

    clusters = {}
    for j in all_jsons:
        if j['label'] not in clusters:
            clusters[j['label']] = 0
        clusters[j['label']] += 1
    
    n_programs = list(clusters.values())[0]

    
    for v in clusters.values():
        assert (
            v == n_programs
        ), "The number of examples for each label should be the same"
    
    print(f"There are {len(clusters)} clusters, each with size {n_programs} ")

if __name__ == "__main__":
    fire.Fire(main)