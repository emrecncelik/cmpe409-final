from __future__ import annotations
from collections import defaultdict


class NGramModel:
    def __init__(
        self,
        n: int = 2,
        padding: bool = True,
        char_level: bool = False,
        bos_token: str = "<s>",
        eos_token: str = "<\s>",
        unk_token: str = "<unk>",
    ) -> None:
        self.n = n
        self.padding = padding
        self.char_level = char_level
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self._vocab = None
        self._token2idx = {}
        self._idx2token = {}
        self._ngram_counts = {}

    @property
    def vocab(self):
        return self._vocab

    @property
    def token2idx(self):
        return self._token2idx

    @property
    def idx2token(self):
        return self._idx2token

    @staticmethod
    def apply_padding(
        sequence: list[str],
        n: int = 2,
        bos_token: str = "<s>",
        eos_token: str = "<\s>",
    ):
        return [bos_token] * (n - 1) + sequence + [eos_token] * (n - 1)

    @staticmethod
    def ngramize(
        sequence: list[str],
        n: int = 2,
        padding: bool = True,
        char_level: bool = False,
        bos_token: str = "<s>",
        eos_token: str = "<\s>",
    ):
        if not char_level:
            if padding:
                sequence = NGramModel.apply_padding(sequence, n, bos_token, eos_token)
            return list(zip(*[sequence[i:] for i in range(n)]))
        else:
            if padding:
                sequence = [
                    NGramModel.apply_padding(list(el), n, bos_token, eos_token)
                    for el in sequence
                ]
            ngrams = []
            for el in sequence:
                ngrams.extend(list(zip(*[el[i:] for i in range(n)])))
            return ngrams

    def train(self, sequences: list[list[str]]):
        if not self.char_level:
            ngrams = []
            if self._vocab is None:
                self._vocab = defaultdict(lambda: 0)

            for sequence in sequences:
                ngrams.append(
                    self.ngramize(
                        sequence,
                        self.n,
                        self.padding,
                        self.char_level,
                        self.bos_token,
                        self.eos_token,
                    )
                )
                for el in sequence:
                    self._vocab[el] += 1
                    idx = len(self._token2idx)
                    self._token2idx[el] = idx
                    self._idx2token[idx] = el

            if self._ngram_counts is None:
                self._ngram_counts
        else:
            pass
