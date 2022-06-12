import os
from itertools import chain

import pandas as pd
from loguru import logger
from nltk.tokenize import sent_tokenize, word_tokenize

from spellchecker.data_collection import collect_from_wiki
from spellchecker.ngrams import BigramLM
from spellchecker.preprocessing import Preprocessor, turkish_lower


MODEL_DIR = os.getenv(
    "MODEL_DIR", "/home/emrecan/workspace/school/2022/cmpe409/final/model_test"
)
DATA_PATH = os.getenv(
    "DATA_PATH", "/home/emrecan/workspace/school/2022/cmpe409/final/dataset.csv"
)

# Collect data
# data = collect_from_wiki(100000)
# data.to_csv(DATA_PATH)

data = pd.read_csv(DATA_PATH)
data = data["content"].tolist()
data = list(chain.from_iterable([sent_tokenize(t) for t in data if isinstance(t, str)]))

# Apply preprocessing
preprocessor = Preprocessor(
    ["lowercase", "normalize_i", "tokenize", "punctuations", "numbers"]
)
data = [list(t) for t in preprocessor(data)]

# Train language model
lm = BigramLM()
lm.train(data)
lm.serialize(MODEL_DIR)  # Save

# Print bigram table
example = word_tokenize(
    turkish_lower(
        "Trakya'dan Anadolu içlerine göçen Frigler daha sonra Misyalılar Kocaeli bölgesinin ilk yerleşik halkı oldukları kabul edilir"
    )
)
indexes = [lm.token2idx[t] for t in example]
logger.info(
    pd.DataFrame(
        lm.bigram_counts[indexes].toarray()[:, indexes], index=example, columns=example
    )
)
