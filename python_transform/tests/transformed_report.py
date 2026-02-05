import json

file = "../output/python_train_1_transformed.jsonl"
report = "./transformation_report.txt"

with open(file, "r") as f:
    json_lines = f.read().splitlines()
    objs = []
    for i, json_line in enumerate(json_lines[:10]):
        objs.append(json.loads(json_line))

    with open(report, "a") as report_f:
        for i, obj in enumerate(objs):
            report_f.write(f"----------{i}-------------\n")
            s = f"original: \n {obj['code']}, \ntransformed: {obj['transformed']}"
            report_f.write(s)
            report_f.write("\n----------end-------------\n\n")
