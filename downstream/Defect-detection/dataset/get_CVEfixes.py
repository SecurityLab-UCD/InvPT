import sqlite3 as lite
from sqlite3 import Error
import random
import fire
import json


def create_connection(db_file: str):
    """
    create a connection to sqlite3 database
    """
    conn = None
    try:
        conn = lite.connect(db_file, timeout=10)  # connection via sqlite3
        # engine = sa.create_engine('sqlite:///' + db_file)  # connection via sqlalchemy
        # conn = engine.connect()
    except Error as e:
        print(e)
    return conn


def get_file_changes_from_sql(cursor: lite.Cursor, language: str):
    cursor.execute(
        """
        SELECT c.repo_url, COUNT(fc.file_change_id) AS file_change_count
        FROM file_change fc
        JOIN commits c ON fc.hash = c.hash
        JOIN repository r ON c.repo_url = r.repo_url
        WHERE fc.code_before IS NOT NULL
        AND fc.code_after IS NOT NULL
        AND fc.programming_language = ?
        GROUP BY c.repo_url
    """,
        (language,),
    )

    repo_file_counts = cursor.fetchall()
    random.shuffle(repo_file_counts)

    total_file_changes = sum(count for _, count in repo_file_counts)
    target_train = total_file_changes * 0.8
    target_val = total_file_changes * 0.1
    target_test = total_file_changes * 0.1

    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # assign repos to train/val/test while keeping the ratio of samples
    for repo, count in sorted(repo_file_counts, key=lambda x: -x[1]):
        remaining = {
            "train": target_train - counts["train"],
            "val": target_val - counts["val"],
            "test": target_test - counts["test"],
        }
        best_split = max(remaining, key=remaining.get)
        splits[best_split].append(repo)
        counts[best_split] += count

    def get_file_change_for_repos(repos):
        if not repos:
            return []
        placeholders = ",".join(["?"] * len(repos))
        query = f"""
            SELECT fc.code_before, fc.code_after, fc.programming_language, c.repo_url
            FROM file_change fc
            JOIN commits c ON fc.hash = c.hash
            WHERE fc.code_before IS NOT NULL
            AND fc.code_after IS NOT NULL
            AND fc.programming_language = 'Python'
            AND c.repo_url IN ({placeholders})
        """
        cursor.execute(query, repos)
        return cursor.fetchall()

    # Step 5: Fetch the actual file_change entries
    train_file_changes = get_file_change_for_repos(splits["train"])
    val_file_changes = get_file_change_for_repos(splits["val"])
    test_file_changes = get_file_change_for_repos(splits["test"])

    print(
        f"Train: {len(train_file_changes)}, Val: {len(val_file_changes)}, Test: {len(test_file_changes)}"
    )

    return train_file_changes, val_file_changes, test_file_changes


def write_data_to_jsonl(file_changes: str, file_name: str, curr_idx: int):
    with open(file_name, "w") as f:
        for code_before, code_after, _, _ in file_changes:
            js_fixed = {"idx": curr_idx, "func": code_after, "target": 0}
            curr_idx += 1
            js_defect = {"idx": curr_idx, "func": code_before, "target": 1}
            curr_idx += 1
            f.write(json.dumps(js_fixed) + "\n")
            f.write(json.dumps(js_defect) + "\n")
    return curr_idx


def main(data_path_sql: str, language: str):
    """
    Args:
        data_path_sql: path to the CVE fixes sql file
        language: language of the code
    Goal:
        Convert the CVE fixes sql file to jsonl file for data split into train, val, test
    """
    conn = create_connection(data_path_sql)
    cursor = conn.cursor()

    train_file_changes, val_file_changes, test_file_changes = get_file_changes_from_sql(
        cursor, language
    )

    curr_idx = write_data_to_jsonl(train_file_changes, "train.jsonl", 0)
    curr_idx = write_data_to_jsonl(val_file_changes, "valid.jsonl", curr_idx)
    write_data_to_jsonl(test_file_changes, "test.jsonl", curr_idx)


if __name__ == "__main__":
    fire.Fire(main)
