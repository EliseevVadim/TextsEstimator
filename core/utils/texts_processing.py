import os
import pandas as pd


def reply_is_valid(reply: str):
    return reply and not reply.isspace()


def create_input_dataframe(data_path: str) -> pd.DataFrame:
    data = []
    file_number = 1
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            name = " ".join(filename.split(' ')[:-2])
            content = file.read()
            last_updated = filename.split(' ')[-1].split('.')[0]
            file_size = os.path.getsize(file_path)
            rounded_size_kb = round(file_size / 1024)
            data.append({'id': file_number,
                         'filename': name,
                         'content': content,
                         'size': file_size,
                         'rounded_size_kb': rounded_size_kb,
                         'last_updated': last_updated})
            file_number += 1

    return pd.DataFrame(data)
