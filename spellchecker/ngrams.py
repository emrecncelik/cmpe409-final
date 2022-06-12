from __future__ import annotations
from collections import Counter
from tqdm import tqdm
import os
from loguru import logger
import numpy as np
import pandas as pd
import pickle
from scipy import sparse


class BigramLM:
    def __init__(
        self, bos_token: str = "<s>", eos_token: str = "<\s>", unk_token: str = "<unk>"
    ) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self._token2idx = None
        self._idx2token = None
        self.bigram_counts = None
        self.token_counts = None

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
        logger.info("Creating indexes...")
        self._token2idx = {
            token: i
            for token, i in zip(self.token_counts.keys(), range(len(self.token_counts)))
        }
        self._idx2token = {i: token for token, i in self.token_counts.items()}

        # Create bigram count matrix
        self.bigram_counts = sparse.lil_matrix(
            (len(self.token_counts), len(self.token_counts))
        )

        # Fill bigram matrix by counting bigrams
        logger.info("Filling bigram count matrix...")
        for sentence_bigrams in tqdm(bigrams):
            for bigram in sentence_bigrams:
                self.bigram_counts[
                    self._token2idx[bigram[0]], self._token2idx[bigram[1]]
                ] += 1

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
            self.bigram_counts[self._token2idx[bigram[0]], self._token2idx[bigram[1]]]
            + 1
        ) / (self.token_counts[bigram[0]] + len(self.token2idx))

    def serialize(self, model_dir: str):
        to_save = {
            "token_counts": self.token_counts,
            "bigram_counts": self.bigram_counts,
            "token2idx": self._token2idx,
            "idx2token": self._idx2token,
        }

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for name, obj in to_save.items():
            logger.info(f"Saving {name}")
            with open(os.path.join(model_dir, name + ".bin"), "wb") as f:
                pickle.dump(obj, f)

    @classmethod
    def from_pretrained(cls, model_dir: str):
        obj = cls()
        to_load = {
            "token_counts": None,
            "bigram_counts": None,
            "token2idx": None,
            "idx2token": None,
        }

        for name in to_load.keys():
            with open(os.path.join(model_dir, name + ".bin"), "rb") as f:
                to_load[name] = pickle.load(f)

        obj.token_counts = to_load["token_counts"]
        obj.bigram_counts = to_load["bigram_counts"]
        obj._token2idx = to_load["token2idx"]
        obj._idx2token = to_load["idx2token"]

        return obj
