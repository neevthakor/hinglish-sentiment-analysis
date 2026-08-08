"""
preprocess.py
-------------
Phase 2 — Data Preprocessing.

Reusable, side-effect-free text/dataframe cleaning functions used by the
Phase 2 pipeline (notebooks/02_preprocessing.py) and by every later phase
(feature engineering, modeling) that needs to clean raw Hinglish tweets
the exact same way.

Design notes
~~~~~~~~~~~~
The corpus (see reports/phase1_data_understanding.md) is code-mixed
Hinglish social-media text (Hindi written in Roman script + English),
already lowercase and largely free of raw URLs/@mentions/#hashtags in
this particular dump — but the functions below are written generically
so they work correctly if/when dirtier raw text (with URLs, mentions,
hashtags, HTML) is fed through this same pipeline in a future phase or
a different data source.

Sentiment-bearing signals we deliberately PRESERVE:
    - Emoji (😂, ❤, 👍 ...) — strong, cheap sentiment signal in social
      media text. We do NOT strip them during punctuation removal.
    - The words carried by a hashtag (e.g. "#modiwins" -> "modiwins" /
      "modi wins") — we strip the "#" but keep the word instead of
      deleting the whole token, since hashtags are often sentiment-
      bearing in political tweets.
    - Devanagari / non-ASCII scripts, in case any slip into the corpus.

Signals we deliberately REMOVE (noise, not sentiment):
    - URLs, HTML tags/entities, @mentions (the handle itself rarely
      carries sentiment).
    - ASCII punctuation (kept minimal impact since emoji are spared).
    - The literal string "nan" left over from an upstream bug where a
      missing value was stringified and concatenated into the text
      (found in ~44% of training rows — see phase2_preprocessing.md).
"""

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# COMPILED REGEX PATTERNS (compiled once at import time for speed)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
# ASCII punctuation only — emoji, Devanagari, and other unicode letters are
# untouched so sentiment-bearing symbols survive cleaning.
PUNCTUATION_PATTERN = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
WHITESPACE_PATTERN = re.compile(r"\s+")
# Upstream data-quality artifact: a stringified missing value ("nan") that
# got concatenated onto the end (mostly) of many rows. Matched as a
# standalone word so we never touch a genuine word that merely contains
# "nan" as a substring (e.g. "ananya").
STRAY_NAN_PATTERN = re.compile(r"\bnan\b", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# ATOMIC TEXT-LEVEL CLEANING FUNCTIONS
# ---------------------------------------------------------------------------
def to_lowercase(text: str) -> str:
    """Lowercase text (safe no-op for already-lowercase Hinglish tokens)."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Strip http(s):// and www. links."""
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags (e.g. <br>) and common HTML entities (&amp; ...)."""
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    return text


def remove_mentions(text: str) -> str:
    """Strip @username mentions entirely (the handle rarely carries sentiment)."""
    return MENTION_PATTERN.sub(" ", text)


def handle_hashtags(text: str) -> str:
    """
    Handle hashtags "appropriately": drop the '#' character but KEEP the
    word that follows it, since hashtag content is frequently
    sentiment-bearing in political/social tweets (e.g. "#shameful" should
    contribute the word "shameful", not disappear entirely).
    """
    return HASHTAG_PATTERN.sub(r"\1", text)


def remove_stray_nan_tokens(text: str) -> str:
    """Remove the literal 'nan' artifact token described in the module docstring."""
    return STRAY_NAN_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """
    Strip ASCII punctuation only. Emoji and non-ASCII characters are left
    untouched so sentiment-bearing symbols are preserved.
    """
    return PUNCTUATION_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and strip the ends."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode to NFC form. This keeps emoji and Devanagari script
    intact while ensuring visually-identical characters compare equal
    (important for Hinglish text that mixes scripts/encodings).
    """
    return unicodedata.normalize("NFC", text)


def clean_hinglish_text(text) -> str:
    """
    Full single-string cleaning pipeline, applied in an order chosen so
    each step operates on the cleanest possible input:

        1. Handle non-string / missing input -> ""
        2. Unicode normalization
        3. Lowercase
        4. Remove URLs
        5. Remove HTML tags/entities
        6. Remove @mentions
        7. Handle hashtags (strip '#', keep the word)
        8. Remove the stray 'nan' artifact token
        9. Remove ASCII punctuation (emoji preserved -> sentiment kept)
        10. Normalize whitespace

    This is the function every other phase should import and reuse
    instead of re-implementing text cleaning.
    """
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_mentions(text)
    text = handle_hashtags(text)
    text = remove_stray_nan_tokens(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# DATAFRAME-LEVEL CLEANING FUNCTIONS
# ---------------------------------------------------------------------------
def remove_duplicate_records(df: pd.DataFrame, subset=("text",)) -> pd.DataFrame:
    """
    Drop fully-duplicated rows first, then rows with duplicate `text`
    (keeping the first occurrence), since two rows with identical text but
    different uid/label are still redundant training signal.
    """
    subset = list(subset)
    before = len(df)

    df = df.drop_duplicates(keep="first")
    df = df.drop_duplicates(subset=subset, keep="first")

    removed = before - len(df)
    print(f"  Removed {removed} duplicate record(s) (full + text-only duplicates).")
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with a missing `text` or missing `label` — there is no safe
    way to impute either for a supervised sentiment-classification task,
    so we discard rather than guess.
    """
    before = len(df)

    missing_text = df["text"].isnull().sum()
    missing_label = df["label"].isnull().sum() if "label" in df.columns else 0

    df = df.dropna(subset=["text"])
    if "label" in df.columns:
        df = df.dropna(subset=["label"])

    removed = before - len(df)
    print(
        f"  Dropped {removed} row(s) with missing values "
        f"(missing text: {missing_text}, missing label: {missing_label})."
    )
    return df.reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Full dataframe-level preprocessing pipeline:
        1. Handle missing values (drop unusable rows)
        2. Remove duplicate records
        3. Apply the text-cleaning pipeline to every row
        4. Drop rows that became empty strings after cleaning
        5. Remove any duplicates created BY cleaning (e.g. two tweets
           that differed only in a URL now collide)

    Returns a new, reset-index DataFrame; the input is not mutated.
    """
    df = df.copy()

    print("  Step 1/5: Handling missing values...")
    df = handle_missing_values(df)

    print("  Step 2/5: Removing duplicate records (pre-clean)...")
    df = remove_duplicate_records(df, subset=(text_col,))

    print("  Step 3/5: Applying text-cleaning pipeline...")
    df[text_col] = df[text_col].apply(clean_hinglish_text)

    print("  Step 4/5: Dropping rows that became empty after cleaning...")
    before = len(df)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} row(s) that were empty after cleaning.")

    print("  Step 5/5: Removing duplicates newly created by cleaning...")
    df = remove_duplicate_records(df, subset=(text_col,))

    return df


def dataset_stats(df: pd.DataFrame, text_col: str = "text") -> dict:
    """Compute a small set of summary statistics used for before/after reporting."""
    texts = df[text_col].dropna().astype(str)
    word_counts = texts.apply(lambda t: len(t.split()))
    return {
        "rows": len(df),
        "nulls": int(df[text_col].isnull().sum()),
        "duplicate_text": int(df[text_col].duplicated().sum()),
        "avg_words": round(word_counts.mean(), 2) if len(word_counts) else 0.0,
        "vocab_size": len(set(" ".join(texts).split())) if len(texts) else 0,
    }
