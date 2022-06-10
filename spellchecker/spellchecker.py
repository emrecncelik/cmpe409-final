from __future__ import annotations
from typing import Callable
from ngrams import BigramLM


class Spellchecker:
    def __init__(
        self,
        lang_model: BigramLM,
        distance_function: Callable,
        max_edit_distance: int = 3,
    ) -> None:
        self.lang_model = lang_model
        self.distance_function = distance_function
        self.max_edit_distance = max_edit_distance

    def spellcheck(self, text: str) -> str:
        """Steps:

        1) Tokenize text
        For each token
            if token not in vocab or token not punct or token not number
                get n candidates for token
                rank candidates based on ngram model
                return best candidate
            else
                return token

        Args:
            text (str): Text to apply spelling correction

        Returns:
            str: Corrected text
        """
        pass

    def _vocab_check(self, token: str) -> bool:
        pass

    def _get_candidates(self, token: str, n: int = 10) -> list[str]:
        pass

    def _select_candidate(self, candidates: list[str]) -> str:
        pass
