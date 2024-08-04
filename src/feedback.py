from typing import List


def provide_context_feedback(
    user_guess: str,
    original_word: str,
    user_fitness: float,
    original_fitness: float,
    contextual_similarity: float,
    top_words: List[str],
    masked_sentence: str,
    context_model
) -> str:
    """
    Provide feedback on the user's guess in terms of contextual fitness.

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
    contextual_similarity : float
        The similarity score between the user's guessed word and the original
        word.
    top_words : List[str]
        A list of top words that fit well in the context.
    masked_sentence : str
        The sentence with a masked word, represented as '[MASK]'.
    context_model : object
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
        f"Your guess: '{user_guess}' (fitness score: {user_fitness:.2f})\n"
        f"Contextual similarity between your guess and the original word: "
        f"{contextual_similarity:.2f}\n\n"
    )

    if user_fitness > original_fitness:
        feedback += ("Great job! Your word actually fits better in this "
                     "context than the original word.\n")
    elif user_fitness > 0.8 * original_fitness:
        feedback += "Excellent! Your word fits very well in this context.\n"
    elif user_fitness > 0.5 * original_fitness:
        feedback += ("Good try! Your word fits reasonably well in this "
                     "context.\n")
    else:
        feedback += ("Your word might not be the best fit for this context, "
                     "but don't worry! Language learning is a process.\n")

    feedback += "\nTop words that fit well in this context:\n"
    for i, word in enumerate(top_words[:5], 1):
        fitness, _ = context_model.get_word_fitness(masked_sentence, word)
        feedback += f"{i}. {word} (fitness: {fitness:.2f})\n"

    return feedback
