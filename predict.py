import os
import logging
from spellchecker import Spellchecker
from spellchecker.ngrams import BigramLM
from nltk.metrics.distance import edit_distance, jaccard_distance, binary_distance

logging.basicConfig(level="INFO")

MODEL_DIR = os.getenv("MODEL_DIR", "model_test")
DATA_PATH = os.getenv("DATA_PATH", "data_test.csv")

lm = BigramLM.from_pretrained(MODEL_DIR)
spellchecker = Spellchecker(
    lm,
    distance_function=edit_distance,
    deduplicate_chars=3,
    n_candidates=10,
    n_jobs=2,
)

print(spellchecker.spellcheck("Hapsssss cezasına çatırıldı!!!"))
print(spellchecker.spellcheck("Halkk tarafndan kabul gördü."))
