from nltk.tokenize import word_tokenize
import re


def mask_word(sentence, frequency_dict, difficulty):
    """
    Mask a word in a sentence based on difficulty level.

    Parameters
    ----------
    sentence : str
        The sentence to mask a word in.
    frequency_dict : collections.Counter
        Dictionary with word frequencies.
    difficulty : str
        Difficulty level ('easy' or 'hard').

    Returns
    -------
    tuple
        Masked sentence and the original word that was masked.
    """
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word.isalnum()]
    word_frequencies = [frequency_dict.get(word, 0) for word in words]

    # Sort words by their frequencies
    sorted_words = sorted(zip(words, word_frequencies),
                          key=lambda x: x[1], reverse=True)

    if difficulty == 'easy':
        # Mask a common word
        word_to_mask = sorted_words[0][0]
    else:
        # Mask a less common word
        word_to_mask = sorted_words[-1][0]

    masked_sentence = re.sub(r"\b" + word_to_mask + r"\b",
                             '[MASK]', sentence, flags=re.IGNORECASE, count=1)
    return masked_sentence, word_to_mask
