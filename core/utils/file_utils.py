import json
import os


def load_evaluation_backup(path: str) -> dict:
    evaluation_backup = {}
    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f:
            evaluation_backup = json.load(f)
    return evaluation_backup


def save_evaluation_backup(evaluation_data: dict, path: str):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=4)
