import pandas as pd


def merge_meta_and_review_data(
    dest_path,
    review_path=None,
    meta_path=None,
):
    df_review = pd.read_json(review_path, lines=True)
    df_meta = pd.read_json(meta_path, lines=True)

    merged_df = df_review.merge(df_meta, how="inner", on="parent_asin")
    merged_df = merged_df[merged_df["verified_purchase"] is True]
    features = [
        "asin",
        "user_id",
        "description",
        "title",
        "price",
        "overall",
        "rating_number",
        "reviewText_senti",
        "positive_prob",
    ]
    merged_df = merged_df[features]
    merged_df.drop_duplicates(subset=["asin", "user_id"], inplace=True)
    merged_df.to_json(dest_path, orient="records", lines=True)
    return merged_df


if __name__ == "__main__":
    merge_meta_and_review_data(
        dest_path="data/merged_reviews.jsonl",
        review_path="data/cleaned_All_Beaauty.jsonl",
        meta_path="data/cleaned_meta_All_Beaauty.jsonl",
    )
