import json
import csv
from typing import Optional, List


def jsonl_to_csv(
    input_file: str, output_file: str, fieldnames: Optional[List[str]] = None
) -> None:
    """
    Convert a JSONL file to CSV format.

    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output CSV file
        fieldnames (Optional[List[str]]): List of field names for CSV headers. If None, will use first JSON object's keys
    """
    # Read JSONL and get data
    data = []
    with open(input_file, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    if not data:
        return

    # If fieldnames not provided, use keys from first record
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})


if __name__ == "__main__":
    jsonl_to_csv("data\\All_Beauty.jsonl", "data\\All_Beauty.csv")
    jsonl_to_csv("data\\meta_All_Beauty.jsonl", "data\\meta_All_Beauty.csv")
