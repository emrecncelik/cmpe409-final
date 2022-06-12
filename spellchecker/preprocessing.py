from __future__ import annotations
import re
import string
from loguru import logger
from toolz import functoolz
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
from tqdm import tqdm


class Preprocessor:
    def __init__(
        self,
        steps: list[str] = [
            "lowercase",
            "normalize_i",
            "tokenize",
            "punctuations",
            "numbers",
        ],
        punctuations: list[str] | None = None,
        numbers: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        n_jobs: int = 1,
    ) -> None:
        self.steps = steps
        self._punctuations = punctuations
        self._numbers = set(numbers)
        self.n_jobs = n_jobs

        self.step_to_func = {
            "lowercase": self.lowercase,
            "normalize_i": self.normalize_i,
            "tokenize": self.tokenize,
            "punctuations": self.remove_punct,
            "numbers": self.remove_numbers,
        }

    def __call__(self, texts: list[str]) -> list[list[str]]:
        logger.info("Preprocessing...")
        return Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(self.pipeline)(t) for t in tqdm(texts)
        )

    @property
    def punctuations(self):
        return set(
            (
                self._punctuations
                if self._punctuations is not None
                else list(string.punctuation) + ["–", "’", "“", "”"]
            )
        )

    @property
    def numbers(self):
        return set(self._numbers)

    @property
    def pipeline(self):
        steps = [
            self.step_to_func[step] if isinstance(step, str) else step
            for step in self.steps
        ]
        return functoolz.compose_left(*steps)

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def remove_punct(self, text: list[str]):
        return (token for token in text if self.is_not_punctuation(token))

    def remove_numbers(self, text: list[str]):
        return (token for token in text if self.is_not_number(token))

    def is_not_number(self, token: str):
        return is_not_number(token, self.numbers, self.punctuations)

    def is_not_punctuation(self, token: str):
        return is_not_punctuation(token, self.punctuations)

    @staticmethod
    def lowercase(text: str) -> str:
        return turkish_lower(text)

    @staticmethod
    def normalize_i(text: str) -> str:
        return text.replace("i̇", "i")


def is_not_punctuation(token: str, puncts: set[str]) -> bool:
    return not all([ch in puncts for ch in token])


def is_not_number(token: str, nums: set[str], puncts: set[str]) -> bool:
    return not all([ch in puncts or ch in nums for ch in token]) and is_not_punctuation(
        token, puncts
    )


def turkish_lower(text: str) -> str:
    text = re.sub(r"İ", "i", text)
    text = re.sub(r"I", "ı", text)
    text = text.lower()
    return text


def turkish_upper(text: str) -> str:
    text = re.sub(r"i", "İ", text)
    text = text.upper()
    return text


if __name__ == "__main__":
    texts = [
        "merhaba ben emrecan'in ... ! 123",
        "Merhabalar 01293 kljasdlkja 91239123.123.",
        "ALOOOOOOOOOOOO İstanbul 01293 kljasdlkja 91239123.123.",
    ]

    preprocessor = Preprocessor()
    print([list(i) for i in preprocessor(texts)])
