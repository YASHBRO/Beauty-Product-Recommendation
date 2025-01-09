import json
import os
from typing import Dict


def _get_jsonl_headers(file_path: str) -> Dict[str, str]:
    # Read the first line of each JSONL file
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line:
            # Parse the JSON line and get keys
            try:
                data: dict = json.loads(first_line)
                return {key: type(value) for key, value in data.items()}
            except json.JSONDecodeError:
                print(f"Error reading {file_path}: Invalid JSON format")


def extract_column_names(input_folder: str, output_folder: str = "") -> None:
    """
    Extract column names from all JSONL files in a folder and write them to a specified file.

    Args:
        input_folder (str): Path to the folder containing JSONL files
        output_file (str): Path to the output file where column names will be written
    """

    if not output_folder:
        output_folder = input_folder

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        columns = dict()
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            columns = _get_jsonl_headers(file_path)

        # Write all unique column names to their output file
        if len(columns.keys()):
            output_file = os.path.join(output_folder, f"{filename}.features.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for column, data_type in sorted(columns.items()):
                    f.write(f"{column}: {data_type}\n")


if __name__ == "__main__":
    extract_column_names("data")
