import random
import nltk
import os
from data_processing import (
    load_corpus, preprocess_text, calculate_frequency_dict)
from ngram import NgramModel
from masking import mask_word
from user import UserProfile, schedule_review, adjust_difficulty
from feedback import provide_feedback
from tqdm import tqdm


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
              "the directory '{corpus_dir}' exists.")
        return

    corpus_files = [os.path.join(corpus_dir, file) for file in os.listdir(
        corpus_dir) if file.endswith('.txt')]
    if not corpus_files:
        print(f"No corpus files found in '{corpus_dir}'. Please add text "
              "files to this directory.")
        return

    # Load and preprocess the corpus
    corpus = ''
    for file in corpus_files:
        corpus += load_corpus(file) + ' '
    tokens = preprocess_text(corpus)

    # Calculate the frequency dictionary
    frequency_dict = calculate_frequency_dict(tokens)

    # Build n-gram model
    ngram_model = NgramModel(3)  # Using trigrams

    # Process sentences
    sentences = nltk.sent_tokenize(corpus)
    for sentence in tqdm(sentences, desc="Building n-gram model"):
        tokens = preprocess_text(sentence)
        ngram_model.update(tokens)

    random.shuffle(sentences)

    # Initialize user profile
    user_profile = UserProfile()

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

        is_correct = original_word.lower() == user_guess.lower()
        feedback = provide_feedback(ngram_model, masked_sentence, user_guess,
                                    original_word, frequency_dict)
        print(feedback)

        user_sentence = masked_sentence.replace('[MASK]', user_guess)
        perplexity = ngram_model.calculate_perplexity(user_sentence)
        print("Perplexity Score: ", perplexity)

        # Update user profile and adjust difficulty
        user_profile.update_performance(is_correct, perplexity)
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
