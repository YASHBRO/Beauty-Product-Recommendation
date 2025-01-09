import os
from typing import Optional
from data_processing.text_processing import clean_text_pipeline
from models.sentiment_generator import sentiment_generator, sample_balancing
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
from sklearn.model_selection import GridSearchCV


def train(df_path, pickle_storage_path: Optional[str] = None, test_reviews=None):
    """
    Trains LinearSVC model on balanced dataset of Amazon
    params:
    df_path = 'All_Beauty_clean.json.gz' #Path of the dataset
    test_reviews = None #List of reviews for model testing
    """
    print("Loading Data ...")
    main_df = pd.read_json(df_path, lines=True)
    features = ["text", "rating"]
    temp_df = main_df[features]

    print("Generating Sentiment ...")
    sentiment = sentiment_generator(temp_df["rating"])

    print("Sampling data ...")
    df = sample_balancing(pd.concat([temp_df, sentiment], axis=1))

    print("Text Processing Pipeline ...")
    df["text"] = df["text"].apply(clean_text_pipeline)

    print("Vectorizing ...")
    ngram_vectorizer = HashingVectorizer(n_features=2**16, ngram_range=(1, 3))

    X = ngram_vectorizer.transform(df["text"])
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Parameter tuning ...")
    param_grid = {"C": [0.1, 0.25, 0.5]}
    svc_model = LinearSVC(max_iter=2000, dual=False)
    grid_search = GridSearchCV(
        svc_model,
        param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=3,
    )
    grid_search.fit(X_train, y_train)
    print(
        f"Best C = {grid_search.best_params_['C']} | Accuracy = {
            grid_search.best_score_}"
    )

    final_svc = grid_search.best_estimator_

    # """## Sanity testing"""

    # feat_coeff = {word: coeff for word, coeff in zip(ngram_vectorizer.get_feature_names(), final_svm.coef_[1])}

    # pos_neg = sorted(feat_coeff.items(), key=lambda x: x[1], reverse=True)
    print("Testing Final model...")
    if test_reviews is None:
        test_reviews = [
            "This product was okish",
            "I have mixed feelings about this product.",
            "it is not upto mark",
            "great",
            "kinda okay",
        ]

    cleaned_test_reviews = [clean_text_pipeline(item) for item in test_reviews]

    X = ngram_vectorizer.transform(cleaned_test_reviews)

    for item in zip(test_reviews, final_svc.predict(X)):
        print(f"{item[0]} >> {item[1]}")

    if pickle_storage_path is not None:
        svc_pickle_folder = pickle_storage_path
    else:
        svc_pickle_folder = "./pickle_files/svc"

    os.makedirs(svc_pickle_folder, exist_ok=True)

    print("Storing vectorizer and model in pickle file ...")
    ngram_vec_file = f"{svc_pickle_folder}/ngram_vec.pkl"
    pickle.dump(ngram_vectorizer, open(ngram_vec_file, "wb"))

    print("Storing model in pickle file ...")
    final_svc_file = f"{svc_pickle_folder}/final_linear_svc.pkl"
    pickle.dump(final_svc, open(final_svc_file, "wb"))
    # del ngram_vectorizer
    # del final_svc
    print("Done.")
    return (ngram_vectorizer, final_svc)


if __name__ == "__main__":
    cleaned_review_path = os.path.join("data", "cleaned_All_Beauty.jsonl")
    dest_path = os.path.join("models", "pickle_files", "svc")
    train(df_path=cleaned_review_path, pickle_storage_path=dest_path)
