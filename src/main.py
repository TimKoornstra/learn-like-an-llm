import random
import nltk
from data_processing import (
    load_corpus, preprocess_text, calculate_frequency_dict)
from ngram import NgramModel
from masking import mask_word


def main():
    """
    Main function to run the word masking and guessing game.

    This function loads and preprocesses a text corpus, calculates the word
    frequency dictionary, builds an n-gram model, processes sentences, and
    facilitates a word guessing game with the user.
    """
    # Load and preprocess the corpus
    corpus = load_corpus('data/corpus/brown.txt')
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
