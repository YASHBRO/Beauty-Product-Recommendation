from data_processing import data_cleaning, data_merge
from models import linear_svc


def main():
    """
    Main function to run the data cleaning and merging process.
    """
    review_path = "data/All_Beauty.jsonl"
    cleaned_review_path = "data/cleaned_All_Beauty.jsonl"
    meta_review_path = "data/meta_All_Beauty.jsonl"
    cleaned_meta_review_path = "data/cleaned_meta_All_Beauty.jsonl"
    merged_data_path = "data/merged_reviews.jsonl"

    # Clean user reviews data
    data_cleaning.review_clean(
        src_path=review_path,
        dest_path=cleaned_review_path,
    )

    # Clean meta data
    data_cleaning.meta_clean(
        src_path=meta_review_path,
        dest_path=cleaned_meta_review_path,
    )

    # Train Linear SVC model
    linear_svc.train(
        df_path=cleaned_review_path,
    )

    # Merge cleaned data
    data_merge.merge_meta_and_review_data(
        dest_path=merged_data_path,
        review_path=cleaned_review_path,
        meta_path=cleaned_meta_review_path,
    )

    print("Data cleaning and merging completed successfully.")
