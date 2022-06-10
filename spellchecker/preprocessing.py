from __future__ import annotations
import string
from itertools import chain
from nltk.tokenize import word_tokenize, sent_tokenize


class Preprocessor:
    def __init__(
        self,
        steps: list[str] = [
            "sent_tokenize",
            "word_tokenize",
            "punctuations",
            "numbers",
            "lowercase",
        ],
        punctuations: list[str] | None = None,
        numbers: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    ) -> None:
        self.steps = steps
        self._punctuations = punctuations
        self._numbers = set(numbers)

        self.step_to_func = {
            "sent_tokenize": self.sent_tokenize,
            "word_tokenize": self.word_tokenize,
            "punctuations": self.remove_punct,
            "numbers": self.remove_numbers,
            "lowercase": self.lowercase,
        }

    @property
    def punctuations(self):
        set(
            (
                self._punctuations
                if self._punctuations is not None
                else list(string.punctuation) + ["–", "’", "“", "”"]
            )
        )

    @property
    def numbers(self):
        return set(self._numbers)

    def __call__(self, texts: list[str]):
        for step in self.steps:
            if isinstance(step, str):
                func = self.step_to_func[step]
            else:
                func = step

    def tokenize_sentences(self, texts: list[str]):
        return chain.from_iterable([sent_tokenize(t) for t in texts])

    def tokenize_words(self, texts: list[str]):
        return (word_tokenize(t) for t in texts)

    def remove_punct(self, texts: list[str]):
        pass

    def remove_numbers(self, texts: list[str]):
        pass

    def is_number(self, token: str):
        return all(
            [ch in self._punctuations or ch in self._numbers for ch in token]
        ) and not self.is_punctuation(token)

    def is_punctuation(self, token: str):
        return all([ch in self._punctuations for ch in token])
