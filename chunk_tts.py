#!/usr/bin/env python3
"""
Lightweight text chunker for TTS.

Features:
- Paragraph splitting (blank lines as hard boundaries).
- Sentence splitting using regex on . ? !
- For sentences > max_len:
    * First, split by commas (minor punctuation).
    * Then, for still-too-long segments, split by subordinators (when/while/where/with/before/until/...).
    * Finally, fallback splitting by commas and "and"/"or" + hard length limit.
- Dash normalization for TTS:
    * "Asia—particularly" -> "Asia, particularly"
    * "associated-though possibly inaccurately-with" -> "associated, though possibly inaccurately, with"
    * keeps "sweet-tart" and "cross-pollination" type compounds

Output:
- JSON list of dicts: { "chunk": str, "rule": str, "length": int }

Usage:
    python chunk_tts_simple.py input.txt > chunks.json
"""

import argparse
import json
import re
import sys
from typing import List, Dict, Any, Optional


# ---------- CONFIG / CONSTANTS ----------

MAX_LEN_DEFAULT = 200  # soft-ish max chunk length (characters)

# Subordinating markers for clause splitting
SUBORDINATORS = {
    "when",
    "while",
    "where",
    "with",
    "before",
    "until",
    "because",
    "although",
    "though",
    "since",
    "after",
    "as",
    "if",
}

# Right-hand discourse markers that usually indicate a clause break after a dash
DISCOURSE = {
    "however",
    "therefore",
    "moreover",
    "instead",
    "nevertheless",
    "nonetheless",
    "furthermore",
    "meanwhile",
    "particularly",
    "especially",
    "otherwise",
    "similarly",
    "consequently",
    "thus",
    "indeed",
    "additionally",
    "though",
}


# ---------- BASIC HELPERS ----------

def normalize_for_tts(text: str) -> str:
    """
    Preprocess text for TTS.

    1) Handle bracketed asides like:
         "associated-though possibly inaccurately-with"
       by turning "- ... -" into ", ... ,"
    2) Handle dash-like punctuation used as clause breaks:
         "Word—particularly" or "Word-particularly" where RHS is a discourse marker
         -> "Word, particularly"
    3) Convert remaining dashes/em-dashes that are NOT between two
       alphanumeric characters into commas.
       True hyphenated compounds like "sweet-tart" or "cross-pollination"
       (letter-letter around '-') are preserved.
    4) Normalize whitespace.
    """

    # 1) Bracketed asides: - some words -
    def replace_aside(match: re.Match) -> str:
        inner = match.group(1)
        if " " in inner.strip():
            return ", " + inner.strip() + ", "
        else:
            return match.group(0)

    text = re.sub(r"[-–—]([^–—\n]*?)[-–—]", replace_aside, text)

    # 2) Handle "Word-DiscourseWord" / "Word—DiscourseWord" patterns
    def replace_word_dash_discourse(match: re.Match) -> str:
        left = match.group(1)
        dash = match.group(2)
        right = match.group(3)
        if right.lower() in DISCOURSE:
            # "Asia-particularly" -> "Asia, particularly"
            return f"{left}, {right}"
        else:
            return match.group(0)

    text = re.sub(
        r"\b([A-Za-z]+)([-–—])([A-Za-z]+)\b",
        replace_word_dash_discourse,
        text,
    )

    # 3) Handle remaining dash-like characters one by one
    chars = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in "-–—":
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < n else " "

            # If dash is between two alphanumerics, keep as hyphen
            if prev.isalnum() and nxt.isalnum():
                chars.append("-")
            else:
                # Treat as a pause -> comma with space
                if chars and chars[-1] == " ":
                    chars[-1] = ","
                    chars.append(" ")
                elif chars and chars[-1] == ",":
                    if nxt != " ":
                        chars.append(" ")
                else:
                    chars.append(",")
                    chars.append(" ")
        else:
            chars.append(ch)

    text = "".join(chars)

    # 4) Normalize whitespace & commas
    text = re.sub(r"\s+,", ",", text)     # no space before comma
    text = re.sub(r",\s*", ", ", text)    # single space after comma
    text = re.sub(r"\s+", " ", text)      # collapse multiple spaces
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines / double newlines.
    We treat paragraphs as hard boundaries for chunking.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(paragraph: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    Keeps the punctuation at the end of each sentence.
    """
    paragraph = re.sub(r"\s+", " ", paragraph.strip())
    if not paragraph:
        return []

    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    if len(parts) == 1:
        return [paragraph]

    return [p.strip() for p in parts if p.strip()]


# ---------- LONG SENTENCE → CLAUSE SPLITTING ----------

def find_subordinator_positions(text: str) -> List[int]:
    """
    Find character indices in text where subordinators occur, as word boundaries.
    """
    starts: List[int] = []
    for word in SUBORDINATORS:
        for m in re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            starts.append(m.start())
    return sorted(set(starts))


def split_by_subordinators(sentence_text: str, max_len: int) -> List[Dict[str, Any]]:
    """
    Split a sentence/segment into clause chunks using subordinators (when/while/where/...).

    If resulting clauses are still too long, fall back to splitting
    by commas and coordinating conjunctions.
    """
    clause_chunks: List[Dict[str, Any]] = []

    starts = find_subordinator_positions(sentence_text)
    if not starts:
        # No subordinators, fallback to punctuation-based splitting
        return fallback_split_by_commas_and_conjs(sentence_text, max_len)

    if starts[0] != 0:
        starts.insert(0, 0)

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(sentence_text)
        clause = sentence_text[start:end].strip()
        if not clause:
            continue

        if len(clause) <= max_len:
            clause_chunks.append({
                "chunk": clause,
                "rule": "clause_subordinator",
            })
        else:
            clause_chunks.extend(
                fallback_split_by_commas_and_conjs(clause, max_len, base_rule="clause_subordinator")
            )

    return clause_chunks


def split_long_sentence(sentence_text: str, max_len: int) -> List[Dict[str, Any]]:
    """
    Split a long sentence into smaller chunks with the following priority:

    1) Split by commas (minor punctuation).
    2) For segments still > max_len, split by subordinators (when/while/where/...).
    3) For anything still too long, fall back to splitting
       by commas and coordinating conjunctions + hard length limit.
    """
    chunks: List[Dict[str, Any]] = []

    # 1) First, split by commas if any
    comma_positions = [i for i, ch in enumerate(sentence_text) if ch == ","]
    if comma_positions:
        segments: List[str] = []
        prev = 0
        for pos in comma_positions:
            end = pos + 1  # include comma
            seg = sentence_text[prev:end].strip()
            if seg:
                segments.append(seg)
            prev = end
        tail = sentence_text[prev:].strip()
        if tail:
            segments.append(tail)

        for seg in segments:
            if len(seg) <= max_len:
                chunks.append({
                    "chunk": seg,
                    "rule": "comma_first",
                })
            else:
                # 2) For segments still too long, use subordinators (then fallback)
                sub_chunks = split_by_subordinators(seg, max_len)
                chunks.extend(sub_chunks)

        return chunks

    # 2) If no commas at all, fall back to subordinators
    return split_by_subordinators(sentence_text, max_len)


def fallback_split_by_commas_and_conjs(text: str,
                                       max_len: int,
                                       base_rule: str = "length_fallback"
                                       ) -> List[Dict[str, Any]]:
    """
    Last-resort splitting for text that is still too long.

    Strategy:
        - Walk through the text up to max_len.
        - Prefer to cut at:
            1. commas
            2. " and " / " or " (simple heuristic)
        - If no good point found, cut hard at max_len.
    """
    chunks: List[Dict[str, Any]] = []
    start = 0
    length = len(text)

    while start < length:
        # Skip leading spaces
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break

        # If remaining fits, just take it
        if length - start <= max_len:
            segment = text[start:length].strip()
            if segment:
                chunks.append({
                    "chunk": segment,
                    "rule": f"{base_rule}+tail",
                })
            break

        window_end = min(start + max_len, length)
        cut_at: Optional[int] = None
        rule = None

        # 1. Prefer comma within the window
        comma_pos = text.find(",", start, window_end)
        if comma_pos != -1:
            cut_at = comma_pos + 1
            rule = f"{base_rule}+comma_split"

        # 2. If no comma, try " and " / " or " near the end of the window
        if cut_at is None:
            window_slice = text[start:window_end]
            match_pos = None
            for conj in [" and ", " or "]:
                pos = window_slice.rfind(conj)
                if pos != -1 and pos > len(window_slice) * 0.3:
                    match_pos = start + pos
                    break
            if match_pos is not None:
                cut_at = match_pos
                rule = f"{base_rule}+conj_split"

        # 3. If still none, hard cut
        if cut_at is None:
            cut_at = window_end
            rule = f"{base_rule}+length_limit"

        segment = text[start:cut_at].strip()
        if segment:
            chunks.append({
                "chunk": segment,
                "rule": rule,
            })
        start = cut_at

    return chunks


# ---------- PARAGRAPH PROCESSING (SINGLE PASS) ----------

def process_paragraph(paragraph: str, max_len: int) -> List[Dict[str, Any]]:
    """
    Process a single paragraph into chunks (sentences or smaller segments).

    - Paragraph is a hard boundary: no chunk crosses it.
    - For each sentence:
        * If length <= max_len: it's a 'sentence' chunk.
        * Else: split using:
             - comma-first strategy
             - then subordinators
             - then fallback splits.
    """
    chunks: List[Dict[str, Any]] = []
    sentences = split_into_sentences(paragraph)

    for sent_text in sentences:
        sent_text = sent_text.strip()
        if not sent_text:
            continue

        if len(sent_text) <= max_len:
            chunks.append({
                "chunk": sent_text,
                "rule": "sentence",
            })
        else:
            # Very long sentence -> comma-first strategy, then subordinators, then fallback
            clause_chunks = split_long_sentence(sent_text, max_len)
            chunks.extend(clause_chunks)

    return chunks


# ---------- TOP-LEVEL CHUNKING ----------

def chunk_text(text: str,
               max_len: int = MAX_LEN_DEFAULT) -> List[Dict[str, Any]]:
    """
    Full chunking pipeline (no merging):

    1) Normalize text (dash handling, spaces).
    2) Split into paragraphs.
    3) For each paragraph: sentence/clause chunks (single pass).
    4) Add length field.
    """
    normalized = normalize_for_tts(text)
    paragraphs = split_into_paragraphs(normalized)

    all_chunks: List[Dict[str, Any]] = []

    for para in paragraphs:
        para_chunks = process_paragraph(para, max_len)
        all_chunks.extend(para_chunks)

    # Final formatting
    final_chunks: List[Dict[str, Any]] = []
    for ch in all_chunks:
        chunk_text_str = ch["chunk"].strip()
        if not chunk_text_str:
            continue
        final_chunks.append({
            "chunk": chunk_text_str,
            "rule": ch["rule"],
            "length": len(chunk_text_str),
        })

    return final_chunks


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk text into TTS-friendly units (no heavy dependencies, no merging)."
    )
    parser.add_argument("input", help="Path to input text file")
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN_DEFAULT,
        help=f"Maximum characters per chunk (default: {MAX_LEN_DEFAULT})",
    )

    args = parser.parse_args()

    # Read file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    chunks = chunk_text(
        raw_text,
        max_len=args.max_len,
    )

    json.dump(chunks, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
