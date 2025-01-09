import os
from typing import Optional
from data_processing.text_processing import clean_text_pipeline
from models.sentiment_generator import sentiment_generator, sample_balancing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pickle


def train(df_path, pickle_storage_path: Optional[str] = None, test_reviews=None):
    """
    Trains Naïve Bayes model on balanced dataset of Amazon
    params:
    df_path = 'All_Beauty_clean.json.gz' #Path of the dataset
    test_reviews = None #List of reviews for model testing
    """

    print("Loading Data...")
    main_df = pd.read_json(df_path, lines=True)
    features = ["text", "rating"]
    temp_df = main_df[features]

    print("Generating Sentiment...")
    sentiment = sentiment_generator(temp_df["rating"])

    print("Sampling data...")
    df = sample_balancing(pd.concat([temp_df, sentiment], axis=1))

    print("Text Cleaning ...")
    df["text"] = df["text"].apply(clean_text_pipeline)

    print("Vectorizing...")
    count_vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 4))
    tfidf_transformer = TfidfTransformer()

    X = df["text"]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=101
    )
    # tfidf_transformer.fit_transform(count_vectorizer.fit_transform(X_train))
    X_train_vec = tfidf_transformer.fit_transform(
        count_vectorizer.fit_transform(X_train)
    )
    X_test_vec = tfidf_transformer.transform(
        count_vectorizer.transform(X_test))

    X_vec = tfidf_transformer.fit_transform(count_vectorizer.fit_transform(X))

    print("Multinomial Naïve Bayes model Training...")
    model_MultinomialNB = MultinomialNB().fit(X_train_vec, y_train)
    mnb_acc = accuracy_score(y_test, model_MultinomialNB.predict(X_test_vec))
    print(f"Accuracy Score : {mnb_acc}")
    del model_MultinomialNB

    print("Bernoulli Naïve Bayes model Training...")
    model_BernoulliNB = BernoulliNB().fit(X_train_vec, y_train)
    bnb_acc = accuracy_score(y_test, model_BernoulliNB.predict(X_test_vec))
    print(f"Accuracy Score : {bnb_acc}")
    del model_BernoulliNB

    if mnb_acc > bnb_acc:
        print("Training Multinomial Naïve Bayes model as final model")
        final_model = MultinomialNB().fit(X_vec, y)
    else:
        print("Training Bernoulli Naïve Bayes model as final model")
        final_model = BernoulliNB().fit(X_vec, y)

    print("Testing Final model...")
    if test_reviews is None:
        test_reviews = ["good", "okay", "good enough", "kind good", "bad"]
        test_reviews = pd.Series(test_reviews)

    print(f"classes : {final_model.classes_}")
    for item in zip(
        test_reviews,
        final_model.predict_proba(
            tfidf_transformer.transform(
                (count_vectorizer.transform(test_reviews)))
        ),
    ):
        print(f"{item[0]} >> {item[1]}")

    print("Making Pickle files...")
    if pickle_storage_path is not None:
        nb_pickle_folder = pickle_storage_path
    else:
        nb_pickle_folder = "./pickle_files/nb"

    os.makedirs(nb_pickle_folder, exist_ok=True)

    print("Storing CountVectorizer")
    count_vect_file = f"{nb_pickle_folder}/count_vect_file.pkl"
    pickle.dump(count_vectorizer, open(count_vect_file, "wb"))

    print("Storing TfidfTransformer")
    tfidf_vect_file = f"{nb_pickle_folder}/tfidf_vect_file.pkl"
    pickle.dump(tfidf_transformer, open(tfidf_vect_file, "wb"))

    print("Storing Final Model")
    final_nb_file = f"{nb_pickle_folder}/final_nb_file.pkl"
    pickle.dump(final_model, open(final_nb_file, "wb"))
    # del count_vectorizer
    # del tfidf_transformer
    # del final_model
    print("Done.")
    return (count_vectorizer, tfidf_transformer, final_model)


if __name__ == "__main__":
    cleaned_review_path = os.path.join("data", "cleaned_All_Beauty.jsonl")
    dest_path = os.path.join("models", "pickle_files", "nb")
    train(df_path=cleaned_review_path, pickle_storage_path=dest_path)
