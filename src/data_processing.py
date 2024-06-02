from collections import Counter
from nltk.tokenize import word_tokenize


def load_corpus(filepath):
    """
    Load text corpus from a file.

    Parameters
    ----------
    filepath : str
        Path to the text corpus file.

    Returns
    -------
    str
        Content of the text file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and tokenizing.

    Parameters
    ----------
    text : str
        The input text to preprocess.

    Returns
    -------
    list of str
        List of alphanumeric tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    return tokens


def calculate_frequency_dict(tokens):
    """
    Calculate the frequency dictionary of tokens.

    Parameters
    ----------
    tokens : list of str
        List of tokens.

    Returns
    -------
    collections.Counter
        Frequency dictionary of tokens.
    """
    return Counter(tokens)
