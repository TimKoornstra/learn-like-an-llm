import random
from nltk.tokenize import word_tokenize
import re


def mask_word(sentence, frequency_dict, difficulty):
    """
    Mask a word in a sentence based on difficulty level using Zipf's law and randomization.

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
        # Mask a common word, with some randomization
        threshold = int(len(sorted_words) * 0.4)  # Top 40% most common words
        common_words = sorted_words[:threshold]
        unique_words = set([word[0] for word in common_words])
        if len(unique_words) == 0:
            # No common words found, mask any word
            word_to_mask = random.choice(words)

        else:
            word_to_mask = random.choice(list(unique_words))
    elif difficulty == 'medium':
        # Mask a less common word, with some randomization
        upper_threshold = int(len(sorted_words) * 0.7)
        lower_threshold = int(len(sorted_words) * 0.4)
        uncommon_words = sorted_words[lower_threshold:upper_threshold]
        unique_words = set([word[0] for word in uncommon_words])
        if len(unique_words) == 0:
            # No uncommon words found, mask any word
            word_to_mask = random.choice(words)
        else:
            word_to_mask = random.choice(list(unique_words))
    else:
        # Mask a rare word, with some randomization
        threshold = int(len(sorted_words) * 0.7)
        rare_words = sorted_words[threshold:]
        unique_words = set([word[0] for word in rare_words])
        if len(unique_words) == 0:
            # No rare words found, mask any word
            word_to_mask = random.choice(words)
        else:
            word_to_mask = random.choice(list(unique_words))

    masked_sentence = re.sub(r"\b" + word_to_mask + r"\b",
                             '[MASK]', sentence, flags=re.IGNORECASE, count=1)
    return masked_sentence, word_to_mask
