from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk

# Download NLTK stopwords if not already present
nltk.download("stopwords")


def stem_text(text):
    """
    PorterStemmer is used for Stemming.
    """
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])


def remove_stopwords(text, stop_word="english"):
    """
    Removes stopwords from the given text.
    Default stop_word is 'english'.
    """
    eng_stop_words = stopwords.words(stop_word)
    return " ".join([word for word in text.split() if word not in eng_stop_words])


def text_clean(
    text,
    reg_no_space=r"[.;:!\'?,\"()\[\]#]",
    reg_space=r"(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)|(;)|(&amp)",
):
    """
    Removes unwanted punctuations, symbols, and HTML tags.
    Default params:
    - reg_no_space: Regex pattern to remove certain characters without spaces.
    - reg_space: Regex pattern to replace certain symbols with spaces.
    """
    no_space = re.compile(reg_no_space)
    space = re.compile(reg_space)

    def preprocess(txt):
        return " ".join(space.sub(" ", no_space.sub("", txt.lower())).split())

    return preprocess(text)


def clean_text_pipeline(text):
    """
    Applies a full cleaning pipeline: text_clean, removing stopwords, and stemming.
    """
    text = text_clean(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
