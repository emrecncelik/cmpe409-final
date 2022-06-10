from __future__ import annotations

import logging
import pandas as pd
from mediawiki import DisambiguationError, MediaWiki
from nltk.tokenize import word_tokenize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_from_wiki(min_word_count: int, n_pages: int = 2) -> pd.DataFrame:
    # Initialize instance for wikipedia api
    wiki = MediaWiki(lang="tr")
    wiki.user_agent = "cmpe409_assignment1_ec"
    collected_page_names = []
    collected_page_contents = []
    word_count = 0

    # Collect random articles until we exceed min word count
    with tqdm(total=min_word_count) as pbar:
        while word_count < min_word_count:
            try:
                # Get page names
                pages = [p for p in wiki.random(n_pages)]
                collected_page_names.extend(pages)

                # Get page content
                for p in pages:
                    p = wiki.page(wiki.search(p)[-1]).content
                    if p:
                        collected_page_contents.append(p)
                    else:
                        collected_page_contents.append("")
                # Count words
                word_count_temp = len(word_tokenize(" ".join(collected_page_contents)))
                pbar.update(word_count_temp)
                word_count += word_count_temp
            except DisambiguationError:
                pass

    return pd.DataFrame(
        {
            "content": collected_page_contents,
        }
    )
