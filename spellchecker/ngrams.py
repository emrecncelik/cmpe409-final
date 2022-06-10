from __future__ import annotations
from collections import Counter

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BigramLM:
    def __init__(
        self, bos_token: str = "<s>", eos_token: str = "<\s>", unk_token: str = "<unk>"
    ) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

    @property
    def token2idx(self):
        return self._token2idx

    @property
    def idx2token(self):
        return self._idx2token

    def train(self, sentences: list[list[str]]):
        bigrams = []
        counter = Counter()
        for sentence in sentences:
            sent = sentence.copy()
            # Pad sentences with start/end token
            sent.append(self.eos_token)
            sent.insert(0, self.bos_token)
            bigrams.append(list(zip(*[sent[i:] for i in range(2)])))
            counter.update(Counter(sent))

        # Calculate unigram counts
        self.token_counts = dict(counter.most_common(None))
        self.token_counts[self.unk_token] = 1
        # Create vocabulary to access tokens
        self._token2idx = {
            token: i
            for token, i in zip(self.token_counts.keys(), range(len(self.token_counts)))
        }
        self._idx2token = {i: token for token, i in self.token_counts.items()}

        # Create bigram count matrix with simple smoothing
        self.bigram_counts = np.ones(
            shape=(len(self.token_counts), len(self.token_counts))
        )

        # Fill bigram matrix by counting bigrams
        for sentence_bigrams in bigrams:
            for bigram in sentence_bigrams:
                self.bigram_counts[self._token2idx[bigram[0]]][
                    self._token2idx[bigram[1]]
                ] += 1
                logger.debug(
                    f"Bigram count for {bigram}:",
                    self.bigram_counts[self._token2idx[bigram[0]]][
                        self._token2idx[bigram[1]]
                    ],
                )

    def predict_sentence_probability(self, sentence: list[str]):
        # Pad sentence
        sentence.append(self.eos_token)
        sentence.insert(0, self.bos_token)

        # Get bigrams from sentence
        bigrams = list(zip(*[sentence[i:] for i in range(2)]))

        # Calculate bigram probabilities for every bigram in the sentence
        probabilities = list(map(self.predict_bigram_probability, bigrams))
        logger.debug(
            pd.DataFrame(
                data=probabilities,
                index=[" ".join(bi) for bi in bigrams],
                columns=["probability"],
            )
        )
        print()

        # Multiply every bigram count in the sentence
        return np.prod(probabilities)

    def predict_bigram_probability(self, bigram: tuple(str)):
        # Replace with unk token if token not in vocab
        bigram = [tok if tok in self.token_counts else self.unk_token for tok in bigram]

        # Calculate P(W_(n-1) | W_n)
        return (
            self.bigram_counts[self._token2idx[bigram[0]]][self._token2idx[bigram[1]]]
            / self.token_counts[bigram[0]]
        )

    def get_bigram_count_df(self):
        return pd.DataFrame(
            data=self.bigram_counts,
            index=self.token_counts.keys(),
            columns=self.token_counts.keys(),
        )
