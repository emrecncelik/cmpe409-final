from __future__ import annotations

import logging
import time
from urllib import request
from urllib.error import URLError

import pandas as pd
from bs4 import BeautifulSoup
from mediawiki import DisambiguationError, MediaWiki
from nltk.tokenize import word_tokenize
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_from_yahoo_news(max_news: int = 50) -> list:
    # Extract news from yahoo
    URL = "https://news.yahoo.com/us/"

    # Initialize Selenium driver
    driver = webdriver.Firefox()
    driver.get(URL)

    # Get html element
    wd_html = driver.find_element(by=By.TAG_NAME, value="html")
    hrefs = []
    for _ in range(20):
        # Scroll down the page
        wd_html.send_keys(Keys.PAGE_DOWN)

        # Finds elements with h3 tags,
        # they mostly contain news articles
        elements = driver.find_elements(by=By.TAG_NAME, value="h3")

        # Get hrefs from found h3 elements
        hrefs_temp = []
        for el in elements:
            try:
                href = el.find_element(by=By.TAG_NAME, value="a").get_property("href")
                logger.info(f"Found href: {href}")
                hrefs_temp.append(href)
            except NoSuchElementException as e:
                logger.error(e)
                continue

        # Filter out hrefs that do not end with html
        hrefs_temp = [href for href in hrefs_temp if href.endswith(".html")]
        hrefs.extend(hrefs_temp)
        hrefs = list(set(hrefs))

        # Quit loop if we exceed max news
        if len(hrefs) >= max_news:
            hrefs = hrefs[:max_news]
            break
        time.sleep(0.2)
    driver.close()

    # Get all texts from found news pages
    texts = []
    logger.info("Getting text from news pages.")
    for href in tqdm(hrefs):
        try:
            response = request.urlopen(href)
            raw = response.read()
            text = BeautifulSoup(raw, "html.parser").get_text()
            texts.append(text)
        except URLError as err:
            logger.error(f"Bad connection, could not get text from {href}")
            logger.error(err)

    return texts


def collect_from_wiki(min_word_count: int, n_pages: int = 2) -> pd.DataFrame:
    # Initialize instance for wikipedia api
    wiki = MediaWiki()
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
            # "name": collected_page_names,
            "content": collected_page_contents,
        }
    )
