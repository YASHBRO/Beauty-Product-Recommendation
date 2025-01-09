import pandas as pd


def sentiment_generator(df_ser: pd.Series) -> pd.DataFrame:
    """
    Takes rating Series as input and returns sentiment DataFrame.
    input : Series
    output : DataFrame
    """

    def _senti(x):
        return -1 if x in [1, 2] else 0 if x == 3 else 1

    return df_ser.apply(_senti).to_frame(name="sentiment")


def sample_balancing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes DataFrame as input and returns balanced Dataframe with equal positive, neutral and negative sentiment.
    input : DataFrame
    output : DataFrame
    """
    low_row = df.loc[(df["sentiment"] == -1), "rating"].value_counts().sum()
    neu_row = df.loc[(df["sentiment"] == 0), "rating"].value_counts().sum()
    high_row = df.loc[(df["sentiment"] == 1), "rating"].value_counts().sum()
    sample_nos = min(low_row, neu_row, high_row)
    neg = df.loc[df["sentiment"] == -1].sample(n=sample_nos, random_state=101)
    neu = df.loc[df["sentiment"] == 0].sample(n=sample_nos, random_state=101)
    pos = df.loc[df["sentiment"] == 1].sample(n=sample_nos, random_state=101)
    df = pd.concat([neg, neu, pos], axis=0)
    return df
