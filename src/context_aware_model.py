from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from typing import List, Tuple
import random
import numpy as np


class ContextAwareTextModel:
    def __init__(self, model_name: str = "bert-base-multilingual-uncased")\
            -> None:
        """
        Initialize the ContextAwareTextModel with a specified pre-trained
        model.

        Parameters
        ----------
        model_name : str, optional
            The name of the pre-trained model to use, by default
            "distilbert-base-uncased".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def get_maskable_tokens(self, sentence: str, difficulty: float) \
            -> List[Tuple[int, str]]:
        """
        Get a list of tokens that can be masked based on the difficulty level.

        Parameters:
        -----------
        sentence : str
            The input sentence to tokenize and analyze.
        difficulty : float
            The difficulty level as a float (higher means more difficult).

        Returns:
        --------
        List[Tuple[int, str]]
            A list of tuples containing the token index and the token itself.
        """
        tokens = self.tokenizer.tokenize(sentence, add_special_tokens=True)
        encoded = self.tokenizer.encode(sentence, add_special_tokens=True)

        # Skip [CLS] and [SEP]
        tokens = tokens[1:-1]
        encoded = encoded[1:-1]

        # Filter out unmaskable tokens
        maskable_tokens = [
            (i, token)
            for i, (token, token_id) in enumerate(zip(tokens, encoded))
            if token.isalpha() and not token.startswith('##')
            and token not in self.tokenizer.all_special_tokens
        ]

        if not maskable_tokens:
            return []

        # Sort by token ID to determine the relative difficulty
        maskable_tokens.sort(key=lambda x: encoded[x[0]])

        # Define the difficulty scaling based on the input score
        n = len(maskable_tokens)

        # Clamp difficulty within a reasonable range, if necessary
        difficulty = max(0, min(difficulty, 1.5))

        # Map difficulty to the index space
        mean_index = int(difficulty * (n - 1))  # Map difficulty to index
        std_dev = n * 0.1  # Standard deviation as 10% of token list length

        # Use a Gaussian distribution centered around the mean_index
        selected_index = int(np.clip(np.random.normal(
            loc=mean_index, scale=std_dev), 0, n - 1))
        selected_token = maskable_tokens[selected_index]

        return [selected_token]

    def mask_word(self, sentence: str, difficulty: str) \
            -> Tuple[str, str, int]:
        """
        Mask a word in the sentence based on the given difficulty.

        Parameters:
        -----------
        sentence : str
            The input sentence to mask a word from.
        difficulty : str
            The difficulty level ('easy', 'medium', or 'hard').

        Returns:
        --------
        Tuple[str, str, int]
            A tuple containing the masked sentence, the original word, and the
            index of the masked token.
        """
        maskable_tokens = self.get_maskable_tokens(sentence, difficulty)

        if not maskable_tokens:
            # If no suitable tokens for the given difficulty, choose any
            # maskable token
            maskable_tokens = self.get_maskable_tokens(sentence, 0.5)

        if not maskable_tokens:
            # If still no maskable tokens, return the original sentence
            return sentence, '', -1

        # Select a token to mask
        mask_index, token_to_mask = random.choice(maskable_tokens)

        # Tokenize the sentence to get tokens with their original positions
        encoded_sentence = self.tokenizer.encode_plus(
            sentence, return_offsets_mapping=True)
        # Exclude [CLS] and [SEP]
        offsets = encoded_sentence['offset_mapping'][1:-1]

        # Find the original word using the offset
        start, end = offsets[mask_index]
        original_word = sentence[start:end]

        # Replace the original word with the mask token in the original
        # sentence
        masked_sentence = sentence[:start] + \
            self.tokenizer.mask_token + sentence[end:]

        return masked_sentence, original_word, mask_index

    def get_top_predictions(self, masked_sentence: str, top_k: int = 10) \
            -> List[str]:
        """
        Generate the top K predictions for a masked token in a sentence.

        Parameters
        ----------
        masked_sentence : str
            The input sentence with a masked token (e.g., "[MASK]").
        top_k : int, optional
            The number of top predictions to return, by default 10.

        Returns
        -------
        List[str]
            A list of the top K predicted tokens that can replace the masked
            token.

        Examples
        --------
        >>> get_top_predictions("The quick brown [MASK] jumps over the lazy dog.", top_k=5)
        ['fox', 'dog', 'cat', 'horse', 'rabbit']
        """
        inputs = self.tokenizer(masked_sentence, return_tensors="pt")
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1)\
            .indices[0].tolist()
        return [self.tokenizer.decode([token]) for token in top_k_tokens]

    def calculate_perplexity(self, sentence: str, word: str, mask_index: int) \
            -> float:
        """
        Calculate the perplexity of a sentence with a masked token and compare
        it with the perplexity when a specific word is placed at the masked
        position.

        Parameters
        ----------
        sentence : str
            The input sentence containing a masked token (e.g., "[MASK]").
        word : str
            The word to place at the masked position for comparison.
        mask_index : int
            The index position of the masked token in the input sentence.

        Returns
        -------
        float
            A tuple containing the perplexity of the original masked sentence
            and the perplexity of the sentence with the specified word
            replacing the masked token.

        Examples
        --------
        >>> calculate_perplexity("The quick brown [MASK] jumps over the lazy dog.", "fox", mask_index=3)
        (20.5, 18.3)
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        labels = inputs.input_ids.clone()

        # Calculate perplexity for the original masked sentence
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss.item()
            perplexity_masked = np.exp(loss)

        # Replace the mask with the word and calculate new perplexity
        inputs.input_ids[0, mask_index] = self.tokenizer.convert_tokens_to_ids(
            [word])[0]
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss.item()
            perplexity_word = np.exp(loss)

        return perplexity_masked, perplexity_word

    def calculate_fitness_score(self,
                                perplexity_masked: float,
                                perplexity_word: float) -> float:
        """
        Calculate the fitness score by comparing the perplexity of a masked
        sentence with the perplexity when a specific word is placed at the
        masked position.

        Parameters
        ----------
        perplexity_masked : float
            The perplexity of the original sentence with the masked token.
        perplexity_word : float
            The perplexity of the sentence with the specified word replacing
            the masked token.

        Returns
        -------
        float
            The fitness score, calculated as the ratio of perplexity_masked to
            perplexity_word. A lower score indicates a better fit.

        Examples
        --------
        >>> calculate_fitness_score(20.5, 18.3)
        1.12
        """
        # Lower perplexity is better, so we invert the ratio
        return perplexity_masked / perplexity_word
