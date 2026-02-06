import json
import random
import argparse


def sample_id_json(json_array, percentage=16):
    """
    Randomly samples a given percentage of JSON objects from the input array.

    :param json_array: List of JSON objects
    :param percentage: Percentage of objects to sample (default: 16%)
    :return: List of sampled JSON objects
    """
    pid_dict = {}
    for p in json_array:
        if p["label"] in pid_dict:
            pid_dict[p["label"]].append(p)
        else:
            pid_dict[p["label"]] = [p]

    final_json = []

    for _, value in pid_dict.items():
        sample_size = max(1, int(len(value) * (percentage / 100)))
        final_json.extend(random.sample(value, sample_size))

    return final_json


def load_jsonl(file_path):
    """
    Loads a JSONL file and returns a list of JSON objects.

    :param file_path: Path to the JSONL file
    :return: List of JSON objects
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(file_path, data):
    """
    Saves a list of JSON objects to a JSONL file.

    :param file_path: Path to the output JSONL file
    :param data: List of JSON objects to save
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample a percentage of JSONL entries."
    )
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")
    parser.add_argument(
        "--percentage",
        type=float,
        default=16,
        help="Percentage of entries to sample (default: 16%)",
    )

    args = parser.parse_args()

    json_array = load_jsonl(args.input_file)
    sampled_data = sample_id_json(json_array, args.percentage)
    save_jsonl(args.output_file, sampled_data)

    print(f"Sampled data saved to {args.output_file}")
