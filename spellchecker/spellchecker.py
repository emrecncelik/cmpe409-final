from __future__ import annotations
import re
from loguru import logger
from difflib import SequenceMatcher
from typing import Callable
from spellchecker.ngrams import BigramLM
from spellchecker.preprocessing import (
    Preprocessor,
    turkish_lower,
    turkish_upper,
)
from nltk.metrics.distance import edit_distance
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize


class Spellchecker:
    def __init__(
        self,
        lang_model: BigramLM,
        distance_function: Callable = edit_distance,
        deduplicate_chars: int = 3,
        n_candidates: int = 10,
        n_jobs: int = 2,
    ) -> None:
        self.lang_model = lang_model
        self.distance_function = distance_function
        self.deduplicate_chars = deduplicate_chars
        self.n_candidates = n_candidates
        self.n_jobs = n_jobs

        self.preprocessor = Preprocessor()

        if deduplicate_chars:
            self.deduplication_pattern = re.compile(
                r"(\w)(\1{NUM,})".replace("NUM", str(int(deduplicate_chars) - 1))
            )

    def spellcheck(self, text: str) -> str:
        """Applies spellchecking to given text.

        Follows this algorithm:
            If token is not punct, not number, and not in vocab
                - Deduplicate
                - Apply spellchecking with Ngrams + String distance
                    - Get N candidates from the distance function
                    - Rank candidates based on the Ngram model (Calculate sentence probabilities with each candidate)
                    - Replace the original token with best candidate
                - Match the casing of spellchecked token with original token.

        Args:
            text (str): Text to apply spellchecking on.

        Returns:
            str: Text with corrected spelling.
        """
        original = word_tokenize(text)
        spellchecked = word_tokenize(self.preprocessor.lowercase(text))

        for idx in range(len(original)):
            token_in_vocab = self._token_in_vocab(spellchecked[idx])

            if (
                self.preprocessor.is_not_punctuation(spellchecked[idx])
                and self.preprocessor.is_not_number(spellchecked[idx])
                and not token_in_vocab
            ):
                logger.info(f"Applying spellchecking for token: {spellchecked[idx]}")

                # Deduplication
                spellchecked[idx] = self._apply_deduplication(spellchecked[idx])
                token_in_vocab = self._token_in_vocab(spellchecked[idx])
                if token_in_vocab:
                    spellchecked[idx] = self._apply_case_transfer(
                        original[idx], spellchecked[idx]
                    )
                    continue

                # Ngram + edit distance
                candidates = self._get_candidates(
                    spellchecked[idx],
                )
                logger.info(f"Candidates from the edit distance: {candidates}")
                scores = []
                for candidate, dist in candidates:
                    sentence_temp = spellchecked.copy()
                    sentence_temp[idx] = candidate
                    sentence_probability = self.lang_model.predict_sentence_probability(
                        [
                            tok
                            for tok in sentence_temp
                            if self.preprocessor.is_not_number(tok)
                            and self.preprocessor.is_not_punctuation(tok)
                        ]
                    )
                    scores.append(sentence_probability * (1 / dist))
                ranked_candidates = sorted(
                    zip(candidates, scores), key=lambda tup: tup[1], reverse=True
                )
                logger.info(
                    f"Candidates ranked by the language model: {ranked_candidates}"
                )
                logger.info(f"Best candidate: {ranked_candidates[0][0][0]}")
                spellchecked[idx] = self._apply_case_transfer(
                    original[idx], ranked_candidates[0][0][0]
                )
            else:
                spellchecked[idx] = original[idx]

        spellchecked_str = ""
        for token in spellchecked:
            if self.preprocessor.is_not_punctuation(token):
                spellchecked_str += " " + token
            else:
                spellchecked_str += token

        return spellchecked_str.strip()

    def _token_in_vocab(self, token: str) -> bool:
        """Check if the token is present in the language model's vocab.

        Args:
            token (str): Token to check if present

        Returns:
            bool: True if token exists in vocab
        """
        return token in self.lang_model.token2idx

    def _get_candidates(self, token: str, n: int = 10) -> list[tuple[str, int]]:
        """Get a list of candidates based on the distance to the given
        token. Distance is calculcated by distance function given in the initialization.

        Args:
            token (str): Token to get candidates for.
            n (int, optional): Number of candidates to return. Defaults to 10.

        Returns:
            list[tuple[str, int]]: List of tuples containing candidates at the 0th index
            and distance at 1st index, sorted ascending.
        """
        distances = []

        distances = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(self.distance_function)(token, candidate)
            for candidate in self.lang_model.token2idx.keys()
        )
        distances = zip(self.lang_model.token2idx.keys(), distances)
        distances = sorted(distances, key=lambda tup: tup[1])

        return distances[:n]

    def _dedup_repl(self, matchobj):
        return matchobj.group(0)[0]

    def _apply_deduplication(self, token: str) -> str:
        """Deduplicate repeating chars in a token.
        eg. Helloooooo -> Hello

        Args:
            token (str): Token to apply deduplication.

        Returns:
            str: Deduplicated token.
        """
        token = self.deduplication_pattern.sub(self._dedup_repl, token)
        logger.info(f"Deduplicated: {token}")
        return token

    def _apply_case_transfer(self, token: str, spellchecked_token: str) -> str:
        """Transfers casing of the original token to spellchecked token
        to maintain the original version of the text.

        Args:
            token (str): Original token with correct casing.
            spellchecked_token (str): Spellchecked token with lowercase.

        Returns:
            str: Spellchecked token transformed into original casing.
        """
        if len(token) == len(spellchecked_token):
            token = case_transfer_matching(token, spellchecked_token)
        else:
            token = case_transfer_similar(token, spellchecked_token)
        return token


# From https://github.com/mammothb/symspellpy
def case_transfer_matching(cased_text: str, uncased_text: str) -> str:
    """Transfers the casing from one text to another - assuming that they are
    'matching' texts, alias they have the same length.
    Args:
        cased_text: Text with varied casing.
        uncased_text: Text that is in lowercase only.
    Returns:
        Text with the content of `uncased_text` and the casing of `cased_text`.
    Raises:
        ValueError: If the input texts have different lengths.
    """
    if len(cased_text) != len(uncased_text):
        raise ValueError(
            "'cased_text' and 'uncased_text' don't have the same length, use "
            "case_transfer_similar() instead"
        )

    return "".join(
        [
            turkish_upper(y) if x.isupper() else turkish_lower(y)
            for x, y in zip(cased_text, uncased_text)
        ]
    )


# From https://github.com/mammothb/symspellpy
def case_transfer_similar(cased_text: str, uncased_text: str) -> str:
    """Transfers the casing from one text to another - for similar (not matching)
    text.
    Use `difflib.SequenceMatcher` to identify the different type of changes
    needed to turn `cased_text` into `uncased_text`.
    - For inserted sections: transfer the casing from the prior character. If no
    character before or the character before is the space, transfer the casing
    from the following character.
    - For deleted sections: no case transfer is required.
    - For equal sections: swap out the text with the original, the cased one, a
    otherwise the two are the same.
    - For replaced sections: transfer the casing using
    :meth:`case_transfer_matching` if the two has the same length, otherwise
    transfer character-by-character and carry the last casing over to any
    additional characters.
    Args:
        cased_text: Text with varied casing.
        uncased_text: Text in lowercase.
    Returns:
        Text with the content of `uncased_text` but the casing of `cased_text`.
    Raises:
        ValueError: If `cased_text` is empty.
    """
    if not uncased_text:
        return uncased_text

    if not cased_text:
        raise ValueError("'cased_text' cannot be empty")

    matcher = SequenceMatcher(a=turkish_lower(cased_text), b=uncased_text)
    result = ""

    for tag, ia1, ia2, ib1, ib2 in matcher.get_opcodes():
        if tag == "delete":
            continue
        if tag == "insert":
            # For the first character or space on the left, take the casing from
            # the following character. Else take case the prior character
            ia_ref = ia1 if ia1 == 0 or cased_text[ia1 - 1] == " " else ia1 - 1
            if cased_text[ia_ref].isupper():
                result += turkish_upper(uncased_text[ib1:ib2])
            else:
                result += turkish_lower(uncased_text[ib1:ib2])
        elif tag == "equal":
            # Transfer the text from the cased_text, as anyhow they are equal
            # (without the casing)
            result += cased_text[ia1:ia2]
        else:
            cased_seq = cased_text[ia1:ia2]
            uncased_seq = uncased_text[ib1:ib2]

            if len(cased_seq) == len(uncased_seq):
                result += case_transfer_matching(cased_seq, uncased_seq)
            else:
                # transfer the casing character-by-character and using the last
                # casing to continue if we run out of the sequence
                for cased, uncased in zip(cased_seq, uncased_seq):
                    result += (
                        turkish_upper(uncased)
                        if cased.isupper()
                        else turkish_lower(uncased)
                    )
                # Apply casing from the last character of cased_seq to the rest
                # of the uncased_seq
                if len(cased_seq) < len(uncased_seq):
                    upper = cased_seq[-1].isupper()
                    idx = len(cased_seq)
                    result += "".join(
                        map(
                            turkish_upper if upper else turkish_upper, uncased_seq[idx:]
                        )
                    )
    return result
