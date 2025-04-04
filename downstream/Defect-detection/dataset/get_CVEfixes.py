import sqlite3 as lite
from sqlite3 import Error
import random
import fire
import json

SEED = 888


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


def add_index(cursor: lite.Cursor, conn: lite.Connection):
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_commits_repo_url ON commits(repo_url);"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_hash ON commits(hash);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_change_hash ON file_change(hash);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_change_language ON file_change(programming_language);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_method_change_file_change_id ON method_change(file_change_id);"
    )
    conn.commit()


# check the number of fixed and defect in train, val, test
def report_num_fixed_and_defect(method_changes: list, set_name: str):
    num_fixed, num_defect = 0, 0
    for _, before_change, _ in method_changes:
        if before_change == "True":
            num_defect += 1
        else:
            num_fixed += 1
    print(f"{set_name}: num_fixed: {num_fixed}, num_defect: {num_defect}")


def get_method_change_for_repos(cursor: lite.Cursor, repos: list, language: str):
    if not repos:
        return []
    placeholders = ",".join(["?"] * len(repos))
    query = f"""
        SELECT 
            mc.code,
            mc.before_change,
            c.repo_url
        FROM commits c
        JOIN file_change fc ON fc.hash = c.hash
        JOIN method_change mc on mc.file_change_id = fc.file_change_id
        WHERE fc.programming_language = ?
        AND c.repo_url IN ({placeholders})
        ORDER BY c.repo_url, fc.file_change_id
    """
    cursor.execute(query, (language,) + tuple(repos))
    return cursor.fetchall()


def make_num_fixed_and_num_defect_equal(method_changes: list):
    num_fixed, num_defect = 0, 0
    fixed_method_changes, defect_method_changes = [], []

    for mc in method_changes:
        # mc[i] = code, before_change, repo_url
        if mc[1] == "True":
            num_defect += 1
            defect_method_changes.append(mc)
        else:
            num_fixed += 1
            fixed_method_changes.append(mc)

    if num_fixed > num_defect:
        fixed_method_changes = random.sample(fixed_method_changes, num_defect)
    elif num_fixed < num_defect:
        defect_method_changes = random.sample(defect_method_changes, num_fixed)

    all_method_changes = fixed_method_changes + defect_method_changes
    random.shuffle(all_method_changes)
    return all_method_changes


def split_method_changes(cursor: lite.Cursor, language: str):
    cursor.execute(
        """
        SELECT c.repo_url, COUNT(mc.method_change_id) AS method_change_count
        FROM method_change mc
        JOIN file_change fc ON mc.file_change_id = fc.file_change_id
        JOIN commits c ON fc.hash = c.hash
        JOIN repository r ON c.repo_url = r.repo_url
        WHERE fc.programming_language = ?
        GROUP BY c.repo_url
    """,
        (language,),
    )

    repo_method_change_counts = cursor.fetchall()
    random.shuffle(repo_method_change_counts)

    # Calculate total and targets
    total_method_change_counts = sum(count for _, count in repo_method_change_counts)
    print(f"total method change counts: {total_method_change_counts}")
    target_train = total_method_change_counts * 0.8
    target_val = total_method_change_counts * 0.1
    target_test = total_method_change_counts * 0.1

    # Keep the ratio of method changes while making sure these sets don't share any repos (to avoid data leakage)
    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for repo, count in sorted(repo_method_change_counts, key=lambda x: -x[1]):
        remaining = {
            "train": target_train - counts["train"],
            "val": target_val - counts["val"],
            "test": target_test - counts["test"],
        }
        best_split = max(remaining, key=remaining.get)
        splits[best_split].append(repo)
        counts[best_split] += count

    # get method changes for train, val, test
    train_method_changes = get_method_change_for_repos(
        cursor, splits["train"], language
    )
    val_method_changes = get_method_change_for_repos(cursor, splits["val"], language)
    test_method_changes = get_method_change_for_repos(cursor, splits["test"], language)

    train_method_changes = make_num_fixed_and_num_defect_equal(train_method_changes)
    val_method_changes = make_num_fixed_and_num_defect_equal(val_method_changes)
    test_method_changes = make_num_fixed_and_num_defect_equal(test_method_changes)

    return train_method_changes, val_method_changes, test_method_changes


def write_data_to_jsonl(method_changes: str, file_name: str, curr_idx: int):
    with open(file_name, "w") as f:
        for code, before_change, _ in method_changes:
            js_data = {
                "idx": curr_idx,
                "func": code,
                "target": 1 if before_change == "True" else 0,
            }
            curr_idx += 1
            f.write(json.dumps(js_data) + "\n")
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
    add_index(cursor, conn)

    train_method_changes, val_method_changes, test_method_changes = (
        split_method_changes(cursor, language)
    )

    report_num_fixed_and_defect(train_method_changes, "train")
    report_num_fixed_and_defect(val_method_changes, "val")
    report_num_fixed_and_defect(test_method_changes, "test")

    curr_idx = write_data_to_jsonl(train_method_changes, "train.jsonl", 0)
    curr_idx = write_data_to_jsonl(val_method_changes, "valid.jsonl", curr_idx)
    write_data_to_jsonl(test_method_changes, "test.jsonl", curr_idx)


if __name__ == "__main__":
    random.seed(SEED)
    fire.Fire(main)
