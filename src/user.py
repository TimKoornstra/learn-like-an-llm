import random


class UserProfile:
    def __init__(self):
        self.difficulty = 'easy'
        self.performance_history = []
        self.words_to_review = {}

    def update_performance(self, is_correct, perplexity):
        self.performance_history.append((is_correct, perplexity))
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    def get_average_score(self):
        if not self.performance_history:
            return 0
        return sum(1 for correct, _ in self.performance_history if correct) \
            / len(self.performance_history)

    def get_current_difficulty(self):
        return self.difficulty

    def set_difficulty(self, new_difficulty):
        self.difficulty = new_difficulty

    def add_word_to_review(self, word, review_time):
        self.words_to_review[word] = review_time

    def get_words_to_review(self):
        return [word for word, time in self.words_to_review.items()
                if time <= 0]

    def update_review_times(self):
        for word in self.words_to_review:
            self.words_to_review[word] -= 1


def schedule_review(user_profile, word):
    """
    Schedule a word for review using spaced repetition.
    """
    # Schedule review after 5-15 more questions
    review_time = random.randint(5, 15)
    user_profile.add_word_to_review(word, review_time)


def adjust_difficulty(user_profile):
    """
    Adjust the difficulty based on the user's performance.
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
