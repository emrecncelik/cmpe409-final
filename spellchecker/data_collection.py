from __future__ import annotations

import pandas as pd
from loguru import logger
from mediawiki import DisambiguationError, MediaWiki
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def collect_from_wiki(min_sent_count: int, n_pages: int = 2) -> pd.DataFrame:
    # Initialize instance for wikipedia api
    wiki = MediaWiki(lang="tr")
    wiki.user_agent = "cmpe409_assignment1_ec"
    collected_page_names = []
    collected_page_contents = []
    sent_count = 0

    # Collect random articles until we exceed min word count
    with tqdm(total=min_sent_count) as pbar:
        while sent_count < min_sent_count:
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
                sent_count_temp = sum(
                    len(sent_tokenize(content))
                    for content in collected_page_contents[-len(pages) :]
                )
                pbar.update(sent_count_temp)
                sent_count += sent_count_temp
                print(sent_count)
            except DisambiguationError:
                pass

    return pd.DataFrame(
        {
            "content": collected_page_contents,
        }
    )
