# File: backend/fastapi/data/data_loader.py

import csv
from typing import List, Dict

def load_dataset(file_path: str) -> List[Dict]:
    """
    Loads the TEDx dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict]: Loaded dataset as a list of dictionaries.
    """
    try:
        data = []
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Initialize 'sdg_tags' as an empty list; will be populated later
                row['sdg_tags'] = []
                data.append(row)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        data = []

    return data
