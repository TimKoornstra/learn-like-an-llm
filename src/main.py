import random
import nltk
import os
from data_processing import (
    load_corpus, preprocess_text, calculate_frequency_dict)
from masking import mask_word
from user import UserProfile, schedule_review, adjust_difficulty
from feedback import provide_context_feedback
from context_aware_model import ContextAwareTextModel


def main():
    """
    Main function to run the word masking and guessing game.

    This function loads and preprocesses a text corpus based on the chosen
    language, calculates the word frequency dictionary, builds an n-gram model,
    processes sentences, and facilitates a word guessing game with the user.
    """
    language = input("Choose a language (e.g., 'english', 'spanish', "
                     "'french'): ").strip().lower()
    corpus_dir = f'data/corpus/{language}'
    if not os.path.isdir(corpus_dir):
        print(f"No corpus found for language '{language}'. Please make sure "
              f"the directory '{corpus_dir}' exists.")
        return

    corpus_files = [os.path.join(corpus_dir, file) for file in os.listdir(
        corpus_dir) if file.endswith('.txt')]
    if not corpus_files:
        print(f"No corpus files found in '{corpus_dir}'. Please add text "
              "files to this directory.")
        return

    # Initialize user profile
    user_profile = UserProfile()

    corpus = ''
    for file in corpus_files:
        corpus += load_corpus(file) + ' '
    tokens = preprocess_text(corpus)

    # Calculate the frequency dictionary
    frequency_dict = calculate_frequency_dict(tokens)

    sentences = nltk.sent_tokenize(corpus)
    random.shuffle(sentences)

    context_model = ContextAwareTextModel()

    while True:
        difficulty = user_profile.get_current_difficulty()
        sentence = sentences.pop(0)
        sentence = sentence.strip().replace('\n', '')
        if len(sentence) == 0:
            continue

        masked_sentence, original_word = mask_word(
            sentence, frequency_dict, difficulty)
        print("\nMasked Sentence: ", masked_sentence)
        user_guess = input("Guess the missing word: ")

        original_fitness, top_words = context_model.get_word_fitness(
            masked_sentence, original_word)
        user_fitness, _ = context_model.get_word_fitness(
            masked_sentence, user_guess)
        contextual_similarity = context_model.calculate_contextual_similarity(
            sentence, original_word, user_guess)

        feedback = provide_context_feedback(user_guess, original_word,
                                            user_fitness, original_fitness,
                                            contextual_similarity, top_words,
                                            masked_sentence, context_model)
        print(feedback)

        # Update user profile and adjust difficulty
        user_profile.update_performance(user_fitness)
        new_difficulty = adjust_difficulty(user_profile)
        user_profile.set_difficulty(new_difficulty)

        # Schedule review for this word
        schedule_review(user_profile, original_word)

        print("\nCurrent Difficulty:", user_profile.get_current_difficulty())
        print("Average Score:", user_profile.get_average_score())
        print("Words to Review:", user_profile.get_words_to_review())

        continue_playing = input("\nContinue? (Y/n): ").lower()
        if continue_playing == 'n':
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
