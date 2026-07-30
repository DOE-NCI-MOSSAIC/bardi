"""Property-based invariant tests for the full default regex chain.

Every example drives the production path: a fresh
``PathologyReportRegexSet().get_regex_set()`` applied by ``CPUNormalizer``
(polars' Rust regex engine, ``lowercase=True``) to a whole table of generated
texts. The asserted invariants are defensible from the library code itself:

- never raises; schema and row count preserved; null in -> null out
- no ``\\r``/``\\n``/``\\t`` (rule 1 removes them; no sub_str reinserts them)
- no two consecutive whitespace characters (the final ``\\s{2,}`` collapse in
  pathology_report.py is unconditional and always appended)
- no backslashes (rules 0/3 strip them; nothing reinserts them)
- every ``[A-Z]+`` run is one of the 13 ``*TOKEN`` sub_strs (lowercasing
  happens before substitution, so tokens are the only uppercase source)
- deterministic across two runs

Explored and deliberately NOT asserted (empirically false):
- idempotency — a second pass lowercases the ``*TOKEN`` strings themselves
- nonempty input -> nonempty stripped output — e.g. ``"("`` normalizes to
  ``" "``

Note the consecutive-whitespace check uses the Unicode ``White_Space`` set
(Rust regex ``\\s`` semantics) rather than Python's ``re`` ``\\s``, which
additionally matches ``\\x1c``-``\\x1f``.
"""

import re
import unittest
from typing import List, Optional

import pyarrow as pa
from hypothesis import given, settings
from hypothesis import strategies as st

from bardi.nlp_engineering import CPUNormalizer, PathologyReportRegexSet

from tests.regex_vectors import KNOWN_TOKENS

# Deterministic, CI-friendly hypothesis profile: same examples every run.
settings.register_profile("ci", derandomize=True, max_examples=75, deadline=None)
settings.load_profile("ci")

# Unicode White_Space codepoints — the set matched by Rust regex's `\s`.
RUST_WHITESPACE = frozenset(
    chr(c)
    for c in (
        list(range(0x09, 0x0E))
        + [0x20, 0x85, 0xA0, 0x1680]
        + list(range(0x2000, 0x200B))
        + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    )
)

# Focused alphabet chosen to hit the regex patterns' trigger characters
# (digits, x, punctuation, math symbols, backslash, control whitespace).
FOCUSED_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .-:x/%<>()\\\n\t\r"

texts_unconstrained = st.lists(
    st.one_of(st.none(), st.text(max_size=300)), max_size=20
)
texts_focused = st.lists(
    st.one_of(st.none(), st.text(alphabet=FOCUSED_ALPHABET, max_size=300)), max_size=20
)


def run_default_chain(texts: List[Optional[str]]) -> List[Optional[str]]:
    """Normalize texts through the production path with a fresh regex set
    (get_regex_set and CPUNormalizer both mutate sub_strs in place)."""
    normalizer = CPUNormalizer(
        fields=["text"],
        regex_set=PathologyReportRegexSet().get_regex_set(),
        lowercase=True,
    )
    table = pa.table({"text": pa.array(texts, type=pa.string())})
    result, _ = normalizer.run(table)
    return result.column("text").to_pylist()


class TestRegexChainProperties(unittest.TestCase):
    """Invariants of the full default chain over generated inputs."""

    def check_invariants(self, texts: List[Optional[str]]) -> None:
        outputs = run_default_chain(texts)

        # Row count preserved, and the chain is deterministic
        self.assertEqual(len(outputs), len(texts))
        self.assertEqual(outputs, run_default_chain(texts))

        for text, output in zip(texts, outputs):
            if text is None:
                self.assertIsNone(output, "null input must stay null")
                continue
            self.assertIsNotNone(output, "non-null input must stay non-null")
            for escape_char in ("\r", "\n", "\t"):
                self.assertNotIn(escape_char, output)
            self.assertNotIn("\\", output)
            for first, second in zip(output, output[1:]):
                self.assertFalse(
                    first in RUST_WHITESPACE and second in RUST_WHITESPACE,
                    f"consecutive whitespace in {output!r} (input {text!r})",
                )
            for uppercase_run in re.findall(r"[A-Z]+", output):
                self.assertIn(
                    uppercase_run,
                    KNOWN_TOKENS,
                    f"unexpected uppercase run in {output!r} (input {text!r})",
                )

    @given(texts=texts_unconstrained)
    def test_invariants_unconstrained_text(self, texts):
        self.check_invariants(texts)

    @given(texts=texts_focused)
    def test_invariants_focused_alphabet(self, texts):
        self.check_invariants(texts)

    def test_schema_preserved(self):
        """Extra columns and field names survive normalization untouched."""
        table = pa.table(
            {"text": ["Report 03/10/01: 1.3x0.7 cm."], "id": [42]}
        )
        normalizer = CPUNormalizer(
            fields=["text"],
            regex_set=PathologyReportRegexSet().get_regex_set(),
            lowercase=True,
        )
        result, _ = normalizer.run(table)
        self.assertEqual(result.column_names, ["text", "id"])
        self.assertEqual(result.column("id").to_pylist(), [42])
        self.assertEqual(result.num_rows, 1)


if __name__ == "__main__":
    unittest.main()
