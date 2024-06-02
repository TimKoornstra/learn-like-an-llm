from collections import defaultdict, Counter
from nltk.util import ngrams
import math
from nltk.tokenize import word_tokenize


class NgramModel:
    """
    N-gram model class to handle n-gram generation and probability
    calculations.

    Attributes
    ----------
    n : int
        The number of n-grams.
    ngram_counts : defaultdict(Counter)
        Counts of n-grams.
    context_counts : Counter
        Counts of contexts.
    """

    def __init__(self, n):
        """
        Initialize the NgramModel with the specified n.

        Parameters
        ----------
        n : int
            The number of n-grams.
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()

    def update(self, tokens):
        """
        Update the n-gram counts with a new set of tokens.

        Parameters
        ----------
        tokens : list of str
            List of tokens to update the model with.
        """
        ngrams_list = ngrams(tokens, self.n, pad_left=True, pad_right=True,
                             left_pad_symbol='<s>', right_pad_symbol='</s>')
        for ngram in ngrams_list:
            context = ngram[:-1]
            word = ngram[-1]
            self.ngram_counts[context][word] += 1
            self.context_counts[context] += 1

    def ngram_prob(self, context, word):
        """
        Calculate the probability of a word given a context.

        Parameters
        ----------
        context : tuple
            The context tuple of words.
        word : str
            The target word.

        Returns
        -------
        float
            The probability of the word given the context.
        """
        if self.context_counts[context] == 0:
            return 1e-6  # Smoothing
        if self.ngram_counts[context][word] == 0:
            return 1e-6
        return self.ngram_counts[context][word] / self.context_counts[context]

    def sentence_prob(self, sentence):
        """
        Calculate the probability of a sentence.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        float
            The probability of the sentence.
        """
        tokens = word_tokenize(sentence.lower())
        tokens = [word for word in tokens if word.isalnum()]
        ngrams_list = ngrams(tokens, self.n, pad_left=True, pad_right=True,
                             left_pad_symbol='<s>', right_pad_symbol='</s>')
        prob = 1.0
        for ngram in ngrams_list:
            context = ngram[:-1]
            word = ngram[-1]
            prob *= self.ngram_prob(context, word)
        return prob

    def calculate_perplexity(self, sentence):
        """
        Calculate the perplexity of a sentence.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        float
            The perplexity score of the sentence.
        """
        prob = self.sentence_prob(sentence)
        perplexity = math.pow(1/prob, 1/len(sentence.split()))
        return perplexity
