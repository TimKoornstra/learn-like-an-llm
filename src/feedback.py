def provide_feedback(ngram_model, masked_sentence, user_guess, original_word,
                     frequency_dict, top_n=5):
    """
    Provide detailed feedback to the user.
    """
    # Calculate perplexity for the original sentence and user's guess
    original_sentence = masked_sentence.replace('[MASK]', original_word)
    user_sentence = masked_sentence.replace('[MASK]', user_guess)
    original_perplexity = ngram_model.calculate_perplexity(original_sentence)
    user_perplexity = ngram_model.calculate_perplexity(user_sentence)

    # Generate a list of alternative words and their perplexities
    words = list(frequency_dict.keys())
    word_perplexities = []
    for word in words:
        if word != original_word and word != user_guess:
            alt_sentence = masked_sentence.replace('[MASK]', word)
            perplexity = ngram_model.calculate_perplexity(alt_sentence)
            word_perplexities.append((word, perplexity))

    # Sort alternatives by perplexity (lower is better)
    word_perplexities.sort(key=lambda x: x[1])

    # Find rank of user's guess
    user_rank = next((i + 1 for i, (word, _)
                     in enumerate(word_perplexities) if word == user_guess),
                     None)
    if user_guess == original_word:
        user_rank = 1
    elif user_rank is not None:
        user_rank += 1  # Account for the original word

    # Prepare feedback
    feedback = f"The original word used in this context was '{original_word}'"\
        f"(perplexity: {original_perplexity:.2f}).\n"
    feedback += f"Your guess '{user_guess}' has a perplexity of " \
        f"{user_perplexity:.2f}.\n"

    if user_rank:
        feedback += f"Your guess ranks #{user_rank} among possible words for "\
            "this context.\n"
    else:
        feedback += "Your guess doesn't appear in our list of common words "\
            "for this context.\n"

    feedback += f"\nHere are the top {top_n} words that could fit in this "\
        "context:\n"
    for i, (word, perplexity) in enumerate(word_perplexities[:top_n], 1):
        feedback += f"{i}. '{word}' (perplexity: {perplexity:.2f})\n"

    return feedback
