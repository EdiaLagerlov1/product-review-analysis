"""Sentence generator for product reviews."""
import random
import pandas as pd
from typing import List, Tuple
from src.utils.logger import LoggerMixin


class SentenceGenerator(LoggerMixin):
    """Generate product review sentences for each category."""

    POSITIVE_WORDS = [
        ["amazing", "excellent", "outstanding", "superb", "fantastic"],
        ["quality", "performance", "design", "features", "build"],
        ["exceeded expectations", "highly recommend", "worth every penny", "perfect choice", "great value"],
        ["love it", "impressed", "satisfied", "happy", "delighted"],
        ["fast shipping", "good packaging", "arrived intact", "well made", "durable"]
    ]

    NEUTRAL_WORDS = [
        ["average", "okay", "decent", "acceptable", "reasonable"],
        ["works fine", "does the job", "meets expectations", "adequate", "functional"],
        ["nothing special", "as described", "standard", "basic", "ordinary"],
        ["some flaws", "could improve", "mixed feelings", "both good and bad", "room for improvement"],
        ["fair price", "moderate quality", "typical", "conventional", "normal"]
    ]

    NEGATIVE_WORDS = [
        ["terrible", "awful", "horrible", "poor", "bad"],
        ["broke", "failed", "stopped working", "defective", "damaged"],
        ["disappointed", "frustrated", "unhappy", "dissatisfied", "regret"],
        ["waste of money", "not worth it", "avoid", "do not buy", "terrible purchase"],
        ["cheap materials", "flawed design", "poor construction", "low quality", "unreliable"]
    ]

    def __init__(self, word_range: Tuple[int, int] = (10, 15)):
        """Initialize sentence generator."""
        self.word_range = word_range

    def _generate_varied_sentence(self, word_groups: List[List[str]], sentiment: str) -> str:
        """Generate a varied sentence by randomly combining words."""
        connectors = ["with", "and", "but", "that", "which", "while", "after", "since"]
        intensifiers = ["very", "really", "quite", "extremely", "absolutely", "somewhat", "fairly"]

        words = []
        for group in word_groups:
            chosen = random.choice(group)
            if random.random() > 0.6 and len(words) > 2:
                words.append(random.choice(intensifiers))
            words.extend(chosen.split())

        num_connectors = random.randint(1, 3)
        for _ in range(num_connectors):
            if len(words) > 4:
                pos = random.randint(2, len(words) - 2)
                words.insert(pos, random.choice(connectors))

        min_words, max_words = self.word_range
        while len(words) < min_words:
            pos = random.randint(0, max(1, len(words) - 1))
            words.insert(pos, random.choice(intensifiers))
        if len(words) > max_words:
            words = words[:max_words]

        if len(words) > 3:
            first_three = words[:3]
            random.shuffle(first_three)
            words[:3] = first_three

        sentence = " ".join(words)
        return sentence.capitalize()

    def generate_sentences(self, num_per_category: int) -> pd.DataFrame:
        """Generate sentences for all categories."""
        self.logger.info(f"Generating {num_per_category} sentences per category")
        data = []

        for _ in range(num_per_category):
            num_groups = random.randint(3, 5)
            selected_groups = random.sample(self.POSITIVE_WORDS, num_groups)
            sentence = self._generate_varied_sentence(selected_groups, 'positive')
            data.append({'sentence': sentence, 'category': 'Positive', 'rating': 5})

        for _ in range(num_per_category):
            num_groups = random.randint(3, 5)
            selected_groups = random.sample(self.NEUTRAL_WORDS, num_groups)
            sentence = self._generate_varied_sentence(selected_groups, 'neutral')
            data.append({'sentence': sentence, 'category': 'Neutral', 'rating': 3})

        for _ in range(num_per_category):
            num_groups = random.randint(3, 5)
            selected_groups = random.sample(self.NEGATIVE_WORDS, num_groups)
            sentence = self._generate_varied_sentence(selected_groups, 'negative')
            data.append({'sentence': sentence, 'category': 'Negative', 'rating': 1})

        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} total sentences")
        return df
