import json
import random
import argparse

def sample_label(json_array, percentage=16):
    """
    Randomly samples a given percentage of JSON objects from the input array.
    
    :param json_array: List of JSON objects
    :param percentage: Percentage of objects to sample (default: 16%)
    :return: List of sampled JSON objects
    """
    pids = list(set(p["label"] for p in json_array))
    sample_size = max(1, int(len(pids) * (percentage / 100)))
    sampled_pids = random.sample(pids, sample_size)

    pid_dict = dict()
    for index, value in enumerate(sampled_pids):
        pid_dict[value] = index
    return pid_dict

def sample_id_json(json_array, pid_dict):
    """
    Randomly samples a given percentage of JSON objects from the input array.
    
    :param json_array: List of JSON objects
    :param percentage: Percentage of objects to sample (default: 16%)
    :return: List of sampled JSON objects
    """
    final_json = []
    for p in json_array:
        if p["label"] in pid_dict:
            p["label"] = pid_dict[p["label"]]
            final_json.append(p)

    return final_json

def load_jsonl(file_path):
    """
    Loads a JSONL file and returns a list of JSON objects.
    
    :param file_path: Path to the JSONL file
    :return: List of JSON objects
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(file_path, data):
    """
    Saves a list of JSON objects to a JSONL file.
    
    :param file_path: Path to the output JSONL file
    :param data: List of JSON objects to save
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a percentage of JSONL entries.")
    parser.add_argument("input_file", help="Path to the folder of JSONL files")
    parser.add_argument("--percentage", type=float, default=16, help="Percentage of entries to sample (default: 16%)")
    
    args = parser.parse_args()
    
    train_array = load_jsonl(f"{args.input_file}/old_train.jsonl")
    test_array = load_jsonl(f"{args.input_file}/old_test.jsonl")
    value_array = load_jsonl(f"{args.input_file}/old_valid.jsonl")
    pid_dict = sample_label(train_array, args.percentage)
    save_jsonl(f"{args.input_file}/train.jsonl", sample_id_json(train_array, pid_dict))
    save_jsonl(f"{args.input_file}/test.jsonl", sample_id_json(test_array, pid_dict))
    save_jsonl(f"{args.input_file}/valid.jsonl", sample_id_json(value_array, pid_dict))
    
    print(f"Sampled data saved")