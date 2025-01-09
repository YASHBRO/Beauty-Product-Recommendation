import os
import pandas as pd
import numpy as np

try:
    try:
        from .text_processing import clean_text_pipeline  # type: ignore
    except Exception:
        from data_processing.text_processing import clean_text_pipeline
except ImportError:
    print(
        "Module text_processing not found. Please ensure it is in the correct directory."
    )

    def clean_text_pipeline(text):
        return text  # Dummy function to avoid errors


import json


def _make_hashable(item):
    if isinstance(item, list):
        return tuple(_make_hashable(i) for i in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in item.items()))
    return item


def reviews_clean(src_path, dest_path):
    """
    Cleans Amazon user reviews dataset.

    Parameters:
        src_path (str): Path to the input dataset.
        dest_path (str): Path where cleaned data will be stored.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        # Try reading JSON with lines=True if it's a JSON Lines file
        df = pd.read_json(src_path, lines=True)
    except ValueError as e:
        print(f"Error reading JSON from {src_path}: {e}")
        print("Attempting to read as a regular JSON file...")
        # Try reading it without lines=True
        try:
            with open(src_path, "r") as f:
                data = json.load(f)  # Using json library for more control
            df = pd.DataFrame(data)  # Convert data to DataFrame
        except Exception as e:
            print(f"Error loading file with json module: {e}")
            return None

    features_not_req = ["reviewTime", "style", "image"]
    df.drop(features_not_req, axis=1, inplace=True, errors="ignore")

    # Fill missing values
    to_impute = ["reviewerName", "reviewText", "summary"]  # Text features
    for col in to_impute:
        try:
            df[col] = df[col].fillna("")
        except KeyError:
            print(f"Column {col} not found in the dataset.")

    if "vote" in df.columns:
        df["vote"].fillna(0, inplace=True)  # Most reviews were not voted by anyone

    # Apply text cleaning
    for col in ["reviewText", "summary"]:
        try:
            df[col] = df[col].apply(clean_text_pipeline)
        except KeyError:
            print(f"Column {col} not found in the dataset.")

    for col in df.columns:
        df[col] = df[col].apply(_make_hashable)

    # Remove duplicates and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    df.to_json(dest_path, orient="records", lines=True)
    return df


def meta_clean(src_path, dest_path):
    """
    Cleans Amazon user reviews meta dataset.

    Parameters:
        src_path (str): Path to the input dataset.
        dest_path (str): Path where cleaned data will be stored.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        df = pd.read_json(src_path, lines=True)
    except ValueError as e:
        print(f"Error reading JSON from {src_path}: {e}")
        print("Attempting to read as a regular JSON file...")
        try:
            with open(src_path, "r") as f:
                data = json.load(f)  # Using json library for more control
            df = pd.DataFrame(data)  # Convert data to DataFrame
        except Exception as e:
            print(f"Error loading file with json module: {e}")
            return None

    features_not_req = [
        "category",
        "tech1",
        "fit",
        "tech2",
        "feature",
        "date",
        "image",
        "main_cat",
        "also_buy",
        "rank",
        "also_view",
        "similar_item",
        "details",
    ]
    df.drop(features_not_req, axis=1, inplace=True, errors="ignore")

    # Convert lists to strings to avoid issues with duplicates
    for col in df.select_dtypes(include=[list]).columns:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Replace blanks and empty lists with NaN
    df = df.map(
        lambda x: np.nan if (isinstance(x, list) and len(x) == 0) or x == "" else x
    )

    # Process 'price' column
    df["price"] = df["price"].apply(lambda x: str(x).strip(" $") if x else np.nan)
    df["price"] = df["price"].apply(lambda x: float(x) if x and len(x) <= 6 else np.nan)
    df["price"] = df["price"].astype(float)

    # Process 'description' column
    df["description"] = df["description"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    )
    df["description"].fillna(
        df["title"], inplace=True
    )  # Impute missing descriptions with titles

    # Handle missing values
    if "brand" in df.columns:
        df["brand"].fillna("", inplace=True)
    df.dropna(subset=["title"], inplace=True)  # Drop rows missing title

    # Apply text cleaning
    for col in ["description"]:
        df[col] = df[col].apply(clean_text_pipeline)

    df["price"].fillna(df["price"].mean(), inplace=True)
    df["price"] = df["price"].round(2)

    for col in df.columns:
        df[col] = df[col].apply(_make_hashable)

    # Remove duplicates and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    df.to_json(dest_path, orient="records", lines=True)
    return df


if __name__ == "__main__":
    src_reviews_path = os.path.join("data", "All_Beauty.jsonl")
    dest_reviews_path = os.path.join("data", "cleaned_All_Beauty.jsonl")
    reviews_clean(src_reviews_path, dest_reviews_path)

    src_meta_path = os.path.join("data", "meta_All_Beauty.jsonl")
    dest_meta_path = os.path.join("data", "cleaned_meta_All_Beauty.jsonl")
    meta_clean(src_meta_path, dest_meta_path)
