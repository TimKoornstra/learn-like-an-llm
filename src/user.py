import random
from typing import List, Dict


class UserProfile:
    def __init__(self) -> None:
        """
        Initialize a UserProfile instance.

        Attributes
        ----------
        difficulty : str
            Current difficulty level.
        performance_history : List[float]
            History of user performance scores.
        words_to_review : Dict[str, int]
            Dictionary of words and their respective review times.
        """
        self.difficulty = 'easy'
        self.performance_history: List[float] = []
        self.words_to_review: Dict[str, int] = {}

    def update_performance(self, fitness: float) -> None:
        """
        Update the user's performance history with a new fitness score.

        Parameters
        ----------
        fitness : float
            The fitness score to be added to the performance history.
        """
        self.performance_history.append(fitness)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    def get_average_score(self) -> float:
        """
        Calculate the average performance score.

        Returns
        -------
        float
            The average performance score.
        """
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)

    def get_current_difficulty(self) -> str:
        """
        Get the current difficulty level.

        Returns
        -------
        str
            The current difficulty level.
        """
        return self.difficulty

    def set_difficulty(self, new_difficulty: str) -> None:
        """
        Set a new difficulty level.

        Parameters
        ----------
        new_difficulty : str
            The new difficulty level to be set.
        """
        self.difficulty = new_difficulty

    def add_word_to_review(self, word: str, review_time: int) -> None:
        """
        Add a word to the review list with a specified review time.

        Parameters
        ----------
        word : str
            The word to be added to the review list.
        review_time : int
            The time until the word should be reviewed.
        """
        self.words_to_review[word] = review_time

    def get_words_to_review(self) -> List[str]:
        """
        Get the list of words that need to be reviewed.

        Returns
        -------
        List[str]
            The list of words that need to be reviewed.
        """
        return [word for word, time in self.words_to_review.items()
                if time <= 0]

    def update_review_times(self) -> None:
        """
        Update the review times for all words in the review list.
        """
        for word in self.words_to_review:
            self.words_to_review[word] -= 1


def schedule_review(user_profile: UserProfile, word: str) -> None:
    """
    Schedule a word for review using spaced repetition.

    Parameters
    ----------
    user_profile : UserProfile
        The user profile to which the word will be added for review.
    word : str
        The word to be scheduled for review.
    """
    review_time = random.randint(5, 15)
    user_profile.add_word_to_review(word, review_time)


def adjust_difficulty(user_profile: UserProfile) -> str:
    """
    Adjust the difficulty based on the user's performance.

    Parameters
    ----------
    user_profile : UserProfile
        The user profile whose difficulty will be adjusted.

    Returns
    -------
    str
        The adjusted difficulty level.
    """
    current_difficulty = user_profile.get_current_difficulty()
    average_score = user_profile.get_average_score()

    if average_score > 0.8 and current_difficulty != 'hard':
        return 'hard'
    elif 0.6 <= average_score <= 0.8 and current_difficulty != 'medium':
        return 'medium'
    elif average_score < 0.6 and current_difficulty != 'easy':
        return 'easy'
    else:
        return current_difficulty
