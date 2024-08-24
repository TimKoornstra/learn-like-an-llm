from typing import List, Tuple
from context_aware_model import ContextAwareTextModel


def provide_context_feedback(
    user_guess: str,
    original_word: str,
    user_fitness: float,
    original_fitness: float,
    top_words: List[Tuple[str, float]],
    masked_sentence: str,
    context_model: ContextAwareTextModel
) -> str:
    """
    Provide feedback on the user's guess in terms of contextual fitness using
    perplexity-based scores.

    Parameters
    ----------
    user_guess : str
        The word guessed by the user to replace the masked word.
    original_word : str
        The actual word that was masked in the sentence.
    user_fitness : float
        The fitness score of the user's guessed word in the context.
    original_fitness : float
        The fitness score of the original word in the context.
    top_words : List[Tuple[str, float]]
        A list of top words that fit well in the context, along with their
        fitness scores.
    masked_sentence : str
        The sentence with a masked word, represented as '[MASK]'.
    context_model : ContextAwareTextModel
        The context model used to calculate fitness scores.

    Returns
    -------
    str
        A feedback string detailing the performance of the user's guess in
        context.
    """
    feedback = (
        f"Original word: '{original_word}' (fitness score: "
        f"{original_fitness:.2f})\n"
        f"Your guess: '{user_guess}' (fitness score: {user_fitness:.2f})\n\n"
    )

    # Adjust these thresholds based on your observed fitness score ranges
    if user_fitness > original_fitness:
        feedback += ("Outstanding! Your word fits even better in this context "
                     "than the original word.\n")
    elif user_fitness > 0.9 * original_fitness:
        feedback += ("Excellent! Your word fits extremely well in this "
                     "context.\n")
    elif user_fitness > 0.7 * original_fitness:
        feedback += "Great job! Your word fits very well in this context.\n"
    elif user_fitness > 0.5 * original_fitness:
        feedback += ("Good try! Your word fits reasonably well in this "
                     "context.\n")
    else:
        feedback += ("Your word might not be the best fit for this context, "
                     "but don't worry! Language learning is a process.\n")

    feedback += ("\nLet's look at how your guess compares to the original word"
                 ":\n")
    if user_fitness > original_fitness:
        feedback += ("Your guess actually reduced the sentence's perplexity "
                     "more than the original word!\n")
    elif user_fitness == original_fitness:
        feedback += ("Your guess fits just as well as the original word in "
                     "this context.\n")
    else:
        difference = (original_fitness - user_fitness) / original_fitness * 100
        feedback += (f"The original word reduces the sentence's perplexity by "
                     f"about {difference:.1f}% more than your guess.\n")

    top_words = sorted(top_words, key=lambda x: x[1], reverse=True)
    feedback += "\nOther words that fit well in this context:\n"
    for i, (word, fitness) in enumerate(top_words[:5], 1):
        feedback += f"{i}. {word} (fitness: {fitness:.2f})\n"

    return feedback


def provide_translations(
    original_sentence: str,
    user_sentence: str,
    language_code: str,
    translator
) -> str:
    """
    Translate a sentence to a specified language.

    Parameters
    ----------
    sentence : str
        The sentence to be translated.
    language_code : str
        The language code for the target language.
    translator : object
        The translator object used for translation.

    Returns
    -------
    str
        The translated sentence.
    """

    if language_code == 'en':
        return "Cannot translate to English."

    original_translation = translator.translate(
        original_sentence, src=language_code, dest='en').text
    user_translation = translator.translate(
        user_sentence, src=language_code, dest='en').text

    ret_str = f"Original Sentence (translated): {original_translation}"

    if original_sentence != user_sentence:
        ret_str += f"\nYour Sentence (translated): {user_translation}\n"

    return ret_str
