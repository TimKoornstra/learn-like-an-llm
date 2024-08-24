import random
import nltk
import os
from data_processing import load_corpus
from user import UserProfile, schedule_review, adjust_difficulty
from feedback import provide_context_feedback, provide_translations
from context_aware_model import ContextAwareTextModel
from googletrans import Translator


def main():
    """
    Main function to run the word masking and guessing game.

    This function loads and preprocesses a text corpus based on the chosen
    language, calculates the word frequency dictionary, builds an n-gram model,
    processes sentences, and facilitates a word guessing game with the user.
    """
    language_code = {'english': 'en', 'spanish': 'es', 'french': 'fr'}
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

    # Initialize translator
    translator = Translator()
    language_code = language_code.get(language, 'auto')

    # Initialize user profile
    user_profile = UserProfile()

    corpus = ''
    for file in corpus_files:
        corpus += load_corpus(file) + ' '

    sentences = nltk.sent_tokenize(corpus)
    random.shuffle(sentences)

    model = ContextAwareTextModel()

    while True:
        sentence = sentences.pop(0)

        masked_sentence, original_word, mask_index = model.mask_word(
            sentence, user_profile.get_average_score())
        if mask_index == -1:
            print("Couldn't find a suitable word to mask in this sentence. "
                  "Skipping...")
            continue
        print("\nMasked Sentence: ", masked_sentence)
        user_guess = input("Guess the missing word: ").strip().lower()
        user_sentence = masked_sentence.replace('[MASK]', user_guess)

        perplexity_masked, perplexity_user = model.calculate_perplexity(
            masked_sentence, user_guess, mask_index)
        user_fitness = model.calculate_fitness_score(
            perplexity_masked, perplexity_user)

        _, perplexity_original = model.calculate_perplexity(
            masked_sentence, original_word, mask_index)
        original_fitness = model.calculate_fitness_score(
            perplexity_masked, perplexity_original)

        top_words_with_fitness = []
        for word in model.get_top_predictions(masked_sentence):
            _, perplexity_word = model.calculate_perplexity(
                masked_sentence, word, mask_index)
            fitness = model.calculate_fitness_score(
                perplexity_masked, perplexity_word)
            top_words_with_fitness.append((word, fitness))

        feedback = provide_context_feedback(
            user_guess,
            original_word,
            user_fitness,
            original_fitness,
            top_words_with_fitness,
            masked_sentence,
            model
        )
        print(feedback)

        if language_code != 'en':
            # Translate the sentence to the user's language
            provided_translation = provide_translations(
                sentence, user_sentence, language_code, translator)
            print(provided_translation)

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
