import os

from nltk.metrics.distance import binary_distance, edit_distance, jaccard_distance

from spellchecker import Spellchecker
from spellchecker.ngrams import BigramLM

MODEL_DIR = os.getenv(
    "MODEL_DIR", "/home/emrecan/workspace/school/2022/cmpe409/final/model_test"
)

lm = BigramLM.from_pretrained(MODEL_DIR)
spellchecker = Spellchecker(
    lm,
    distance_function=edit_distance,
    deduplicate_chars=3,
    n_candidates=10,
    n_jobs=2,
)

texts = ["Hapsssss cezasına çatırıldı!!!", "Halkk tarafndan kabul gördü."]

for t in texts:
    print(f"Original: {t}")
    spellchecked = spellchecker.spellcheck(t)
    print(f"Spellchecked: {spellchecked}")
