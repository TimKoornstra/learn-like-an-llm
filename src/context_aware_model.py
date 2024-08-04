from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class ContextAwareTextModel:
    def __init__(self, model_name: str = "bert-base-multilingual-uncased") -> None:
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
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def get_word_fitness(
        self,
        masked_sentence: str,
        word: str,
        top_k: int = 10
    ) -> Tuple[float, List[str]]:
        """
        Calculate the fitness score of a word in a masked sentence and return
        the top-k predicted words.

        Parameters
        ----------
        masked_sentence : str
            The sentence with a masked word, represented as '[MASK]'.
        word : str
            The word to evaluate in the context of the masked sentence.
        top_k : int, optional
            The number of top predicted words to return, by default 10.

        Returns
        -------
        Tuple[float, List[str]]
            A tuple containing the fitness score of the word and a list of the
            top-k predicted words.

        Raises
        ------
        ValueError
            If the mask token logits tensor is empty.
        """
        masked_sentence = masked_sentence.replace(
            "[MASK]", self.tokenizer.mask_token)
        inputs = self.tokenizer(masked_sentence, return_tensors="pt")

        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]

        if mask_token_logits.numel() == 0:
            raise ValueError("Mask token logits tensor is empty.")

        top_k_tokens = torch.topk(
            mask_token_logits, top_k, dim=1).indices[0].tolist()
        top_k_words = [self.tokenizer.decode(
            [token]) for token in top_k_tokens]

        word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        word_logit = mask_token_logits[0, word_id].item()
        max_logit = torch.max(mask_token_logits).item()
        min_logit = torch.min(mask_token_logits).item()
        fitness_score = (word_logit - min_logit) / (max_logit - min_logit)

        return fitness_score, top_k_words

    def calculate_contextual_similarity(
        self,
        sentence: str,
        word1: str,
        word2: str
    ) -> float:
        """
        Calculate the contextual similarity between two words in a sentence.

        Parameters
        ----------
        sentence : str
            The sentence in which the words appear.
        word1 : str
            The first word to compare.
        word2 : str
            The second word to compare.

        Returns
        -------
        float
            The contextual similarity score between the two words.
        """
        embedding1 = self.sentence_model.encode(sentence.replace(word1, word2))
        embedding2 = self.sentence_model.encode(sentence)

        return self.sentence_model.similarity(embedding1, embedding2).item()
