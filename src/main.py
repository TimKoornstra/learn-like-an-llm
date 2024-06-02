import random
import nltk
import os
from data_processing import (
    load_corpus, preprocess_text, calculate_frequency_dict)
from ngram import NgramModel
from masking import mask_word


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
        print(
            f"No corpus found for language '{language}'. Please make sure the "
            f"directory '{corpus_dir}' exists.")
        return

    corpus_files = [os.path.join(corpus_dir, file) for file in os.listdir(
        corpus_dir) if file.endswith('.txt')]
    if not corpus_files:
        print(
            f"No corpus files found in '{corpus_dir}'. Please add text files "
            "to this directory.")
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
    for sentence in sentences:
        tokens = preprocess_text(sentence)
        ngram_model.update(tokens)
    random.shuffle(sentences)

    user_score = 0
    user_correct = 0
    total_sentences = 0
    for sentence in sentences:
        sentence = sentence.strip().replace('\n', '')
        if len(sentence) == 0:
            continue
        masked_sentence, original_word = mask_word(
            sentence, frequency_dict, 'easy')
        print("Masked Sentence: ", masked_sentence)
        user_guess = input("Guess the missing word: ")
        user_sentence = masked_sentence.replace('[MASK]', user_guess)
        perplexity = ngram_model.calculate_perplexity(user_sentence)
        print("Perplexity Score: ", perplexity)
        if original_word.lower() == user_guess.lower():
            print("Correct!")
            user_correct += 1
        else:
            print("Incorrect.")
            print("Original Word: ", original_word)
            print("Original Sentence: ", sentence)
        user_score += perplexity
        total_sentences += 1

        print("\nUser Score: ", user_score / total_sentences)
        print("User Accuracy: ", user_correct / total_sentences)
        print("Total Sentences: ", total_sentences)
        print("-------------------")


if __name__ == "__main__":
    main()
