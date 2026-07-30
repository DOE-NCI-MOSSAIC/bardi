"""Set of functions to support the creation of more complicated test data.

Also provides idempotent ``ensure_*`` fixture helpers used by the test
suite to generate the pickled test DataFrames under ``tests/test_data/``
on first use. All generation is deterministic (seeded) so fixtures are
reproducible across machines.
"""
import random
import string
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

# Anchor all fixture paths to the repo's tests/ directory rather than CWD
TESTS_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = TESTS_DIR / "test_data"

NUM_ROWS = 127


def generate_fake_vocabulary(
    voc_size: int = 100,
    existing_words: Optional[Iterable[str]] = None,
) -> List[str]:
    """Generate a vocabulary of unique fake lowercase words.

    Words are unique within the returned vocabulary and disjoint from
    ``existing_words`` (if provided), so vocabularies built sequentially
    never overlap. This keeps the total vocabulary size across text
    columns exact and deterministic.

    Args:
        voc_size: Number of unique words to generate.
        existing_words: Words that must not appear in the returned
            vocabulary. Pass previously generated vocabularies here to
            keep them disjoint.

    Returns:
        A list of ``voc_size`` unique words (3-8 lowercase ASCII letters
        each), none of which occur in ``existing_words``.
    """
    if existing_words is None:
        existing_words = set()
    fake_vocab = []
    seen = set(existing_words)
    while len(fake_vocab) < voc_size:
        word_length = random.randint(3, 8)
        fake_word = ''.join(random.choices(string.ascii_lowercase,
                                           k=word_length))
        if fake_word not in seen:
            seen.add(fake_word)
            fake_vocab.append(fake_word)
    return fake_vocab


def generate_fake_text(
    vocabulary: List[str],
    min_word_count: int = 30,
    max_word_count: int = 300,
) -> str:
    """Generate a fake text document by sampling words from a vocabulary.

    Args:
        vocabulary: Words to sample from (with replacement).
        min_word_count: Minimum number of words in the document.
        max_word_count: Maximum number of words in the document.

    Returns:
        A single space-joined string of randomly sampled words with a
        length (in words) drawn uniformly from
        ``[min_word_count, max_word_count]``.
    """
    word_count = random.randint(min_word_count, max_word_count)
    fake_text = random.choices(vocabulary, k=word_count)
    fake_text = ' '.join(fake_text)
    return fake_text


def create_mock_data(num_rows: int) -> Dict[str, list]:
    """Create a deterministic mock dataset with ``num_rows`` rows.

    The three text columns are drawn from disjoint vocabularies of
    200/100/300 unique words, giving exactly 600 unique tokens overall
    (the embedding generator tests assert 600 + <pad> + <unk> = 602).

    Args:
        num_rows: Number of rows to generate.

    Returns:
        A mapping of column name to column values, suitable for
        ``pd.DataFrame(...)``. Columns: ``id``, ``state``, ``letter``,
        ``feature_1`` (int), ``feature_2`` (float), ``feature_3``
        (bool), and ``text_1``/``text_2``/``text_3`` (str).

    Note:
        Reseeds the *global* ``random`` and ``numpy.random`` state
        (seed 42) as a side effect, so output is identical on every
        call. Callers relying on prior global RNG state should not
        call this mid-sequence.
    """
    random.seed(42)
    np.random.seed(42)

    fake_vocab1 = generate_fake_vocabulary(voc_size=200)
    fake_vocab2 = generate_fake_vocabulary(voc_size=100,
                                           existing_words=fake_vocab1)
    fake_vocab3 = generate_fake_vocabulary(
        voc_size=300, existing_words=fake_vocab1 + fake_vocab2)

    state_list = ["NY", "NH", "CA", "FL", "NM", "NC", "ID"]
    letter_list = ["A", "B", "C", "D"]
    feature_1 = [1, 2, 3, 4, 5]
    feature_2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    feature_3 = [True, False]

    data = {
        'id': [x for x in range(num_rows)],
        'state': [random.choice(state_list) for _ in range(num_rows)],
        'letter': [random.choice(letter_list) for _ in range(num_rows)],
        'feature_1': [random.choice(feature_1) for _ in range(num_rows)],
        'feature_2': [random.choice(feature_2) for _ in range(num_rows)],
        'feature_3': [random.choice(feature_3) for _ in range(num_rows)],
        'text_1': [generate_fake_text(fake_vocab1) for _ in range(num_rows)],
        'text_2': [generate_fake_text(fake_vocab2) for _ in range(num_rows)],
        'text_3': [generate_fake_text(fake_vocab3) for _ in range(num_rows)]
    }
    return data


def ensure_test_data_dir() -> Path:
    """Create the ``tests/test_data/`` directory if needed.

    The directory is gitignored and holds generated fixtures and
    test scratch files. Its location is anchored to this file (the
    repository layout), not the current working directory.

    Returns:
        Absolute path to ``tests/test_data/``.
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR


def ensure_pipeline_fixture() -> Path:
    """Generate ``pipeline_test_df.pkl`` if it does not already exist.

    Used by ``tests/pipeline_tests.py``. Idempotent: an existing file
    is left untouched.

    Returns:
        Absolute path to the pickled mock DataFrame fixture.
    """
    fixture_path = ensure_test_data_dir() / "pipeline_test_df.pkl"
    if fixture_path.exists():
        return fixture_path
    data_df = pd.DataFrame(create_mock_data(NUM_ROWS))
    data_df.to_pickle(fixture_path)
    return fixture_path


def ensure_embed_gen_fixture() -> Path:
    """Generate ``embed_gen_test_df.pkl`` if it does not already exist.

    Used by ``tests/embedding_generator_tests.py``. The embedding
    generator expects pre-tokenized text columns (lists of strings),
    so the mock text is whitespace-split before pickling. Idempotent:
    an existing file is left untouched.

    Returns:
        Absolute path to the pickled mock DataFrame fixture.
    """
    fixture_path = ensure_test_data_dir() / "embed_gen_test_df.pkl"
    if fixture_path.exists():
        return fixture_path
    data_df = pd.DataFrame(create_mock_data(NUM_ROWS))
    for field in ["text_1", "text_2", "text_3"]:
        data_df[field] = data_df[field].str.split(" ")
    data_df.to_pickle(fixture_path)
    return fixture_path


def ensure_split_fixture() -> Path:
    """Generate ``split_test_df.pkl`` if it does not already exist.

    Used by ``tests/splitter_tests.py``. Idempotent: an existing file
    is left untouched.

    Note:
        Snapshot semantics: the golden ``split_correct`` column is
        produced by running ``CPUSplitter(NewSplit(...))`` itself (with
        the exact parameters used in ``tests/splitter_tests.py``), so
        the splitter tests validate determinism/regressions against
        this snapshot rather than first-time correctness of the split
        algorithm.

    Returns:
        Absolute path to the pickled mock DataFrame fixture, containing
        the mock columns plus the golden ``split_correct`` column.
    """
    fixture_path = ensure_test_data_dir() / "split_test_df.pkl"
    if fixture_path.exists():
        return fixture_path

    import pyarrow as pa
    from bardi.nlp_engineering import CPUSplitter, NewSplit

    data_df = pd.DataFrame(create_mock_data(NUM_ROWS))
    splitter = CPUSplitter(NewSplit(
        split_proportions={'train': 0.7,
                           'test': 0.15,
                           'val': 0.15},
        unique_record_cols=['id'],
        group_cols=['state',
                    'letter'],
        label_cols=None,
        random_seed=42)
    )
    split_data, _ = splitter.run(pa.Table.from_pandas(data_df))
    split_df = split_data.to_pandas()
    split_df = split_df.rename(columns={'split': 'split_correct'})
    split_df.to_pickle(fixture_path)
    return fixture_path


def main() -> None:
    """Generate all test fixtures (no-op for those that already exist)."""
    ensure_pipeline_fixture()
    ensure_embed_gen_fixture()
    ensure_split_fixture()


if __name__ == '__main__':
    main()
