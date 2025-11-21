# Chunking for TTS

This is a mini challenge for chunking text for TTS generation.

In this document, I will show:

1. The given task
2. My solutions
3. How I approach the task (the interesting stuff!)

In this codebase, you will find `chunk_tts.py`, the script containing all the code. It reads an input file and chunks the text into JSON format.

Usage:

```shell
python chunk_tts.py inputs/sample1.tx > outputs/sample1.json
```

```shell
python chunk_tts.py inputs/sample2.txt > outputs/sample2.json
```

## 1. The Given Task

This is the original task provided *(can skip reading)*.

```I want you to break text into chunks, where each chunk is strictly fewer than 200 characters. Your goal is to optimize the text for text-to-speech (TTS) generation.

**Instructions:**

1. Chunking:
Divide the text into logical, meaningful pieces that do not exceed 200 characters each. Ensure each chunk maintains natural phrasing suitable for speech synthesis.

2. Redundancy Removal:
As well as the text chunking, remove any characters or elements that are redundant or unnecessary for speech generation.

3. Explain your reasoning:
    Alongside your output, explain:
    - How you chose the chunk boundaries (e.g., sentence ends, phrase cadence)
    - What specific characters or elements you removed and why
    - How your changes improve the result for TTS

4. What would you have to think about to make this work for other languages than english?``` 

text1 = """Apples are one of the most widely cultivated and consumed fruits in the world. Known for their crisp texture, sweet-tart flavor, and impressive versatility, 
apples have a long and fascinating history that spans cultures, continents, and centuries. Botanically classified as Malus domestica, apples belong to the Rosaceae family, which also includes pears, cherries, and roses.

The domestic apple traces its ancestry to the wild apple species Malus sieversii, native to the mountains of Central Asia—particularly in what is now Kazakhstan. Ancient traders along the Silk Road helped spread apples westward to Europe and eastward to China. Over time, through both natural cross-pollination and human intervention, the apple evolved into the diverse array of cultivars we enjoy today.

Apples were highly prized by ancient civilizations. The Greeks and Romans cultivated them extensively, and they became symbolic in many myths and traditions. In Norse mythology, apples were believed to grant eternal youth. In the biblical tradition, the apple became associated—though possibly inaccurately—with the fruit of the Tree of Knowledge."""


text2 = """when the sun rises over the distant hills and the birds begin to sing their morning songs while the dew still clings to the grass and the breeze carries the scent of blooming flowers through the quiet streets where people slowly start to stir from their sleep and the world gently shifts from night to day with a sense of calm that is fleeting yet beautiful and everything feels suspended in a moment of possibility before the rush of time resumes its usual pace and the responsibilities of life return to fill the hours with motion sound and urgency until the sun once again sinks below the horizon and darkness wraps the earth in stillness once more """
```

## 2. My Solutions

Sample 1 solution:

```
[
  {
    "chunk": "Apples are one of the most widely cultivated and consumed fruits in the world.",
    "rule": "sentence",
    "length": 78
  },
  {
    "chunk": "Known for their crisp texture, sweet-tart flavor, and impressive versatility, apples have a long and fascinating history that spans cultures, continents, and centuries.",
    "rule": "sentence",
    "length": 168
  },
  {
    "chunk": "Botanically classified as Malus domestica, apples belong to the Rosaceae family, which also includes pears, cherries, and roses.",
    "rule": "sentence",
    "length": 128
  },
  {
    "chunk": "The domestic apple traces its ancestry to the wild apple species Malus sieversii, native to the mountains of Central Asia, particularly in what is now Kazakhstan.",
    "rule": "sentence",
    "length": 162
  },
  {
    "chunk": "Ancient traders along the Silk Road helped spread apples westward to Europe and eastward to China.",
    "rule": "sentence",
    "length": 98
  },
  {
    "chunk": "Over time, through both natural cross-pollination and human intervention, the apple evolved into the diverse array of cultivars we enjoy today.",
    "rule": "sentence",
    "length": 143
  },
  {
    "chunk": "Apples were highly prized by ancient civilizations.",
    "rule": "sentence",
    "length": 51
  },
  {
    "chunk": "The Greeks and Romans cultivated them extensively, and they became symbolic in many myths and traditions.",
    "rule": "sentence",
    "length": 105
  },
  {
    "chunk": "In Norse mythology, apples were believed to grant eternal youth.",
    "rule": "sentence",
    "length": 64
  },
  {
    "chunk": "In the biblical tradition, the apple became associated, though possibly inaccurately, with the fruit of the Tree of Knowledge.",
    "rule": "sentence",
    "length": 126
  }
]
```

Sample 2 solution:

```
[
  {
    "chunk": "when the sun rises over the distant hills and the birds begin to sing their morning songs",
    "rule": "clause_subordinator",
    "length": 89
  },
  {
    "chunk": "while the dew still clings to the grass and the breeze carries the scent of blooming flowers through the quiet streets",
    "rule": "clause_subordinator",
    "length": 118
  },
  {
    "chunk": "where people slowly start to stir from their sleep and the world gently shifts from night to day",
    "rule": "clause_subordinator",
    "length": 96
  },
  {
    "chunk": "with a sense of calm that is fleeting yet beautiful and everything feels suspended in a moment of possibility",
    "rule": "clause_subordinator",
    "length": 109
  },
  {
    "chunk": "before the rush of time resumes its usual pace and the responsibilities of life return to fill the hours",
    "rule": "clause_subordinator",
    "length": 104
  },
  {
    "chunk": "with motion sound and urgency",
    "rule": "clause_subordinator",
    "length": 29
  },
  {
    "chunk": "until the sun once again sinks below the horizon and darkness wraps the earth in stillness once more",
    "rule": "clause_subordinator",
    "length": 100
  }
]
```


## 3. How I approach the task

### 3.1. Background, preliminary tests

I begin by reviewing the research and development in the relevant domain. Chunking has been a topic of NLP research for over 30 years. It is an essential part of understanding/emulating natural language.

Some primary approaches to chunking:
- Rule-based systems: lightweight heuristics to efficiently chunk text
- ML models: using pretrained models (i.e. SBERT) to encode or compute embeddings for text chunks
- LLM-based: LLMs can accurately (though inefficiently) chunk text

While state-of-the-art techniques such as ML models and LLM-based chunking can give more sophisticated outputs, they are also more computationally demanding. I had tried solving the task with some of these methods and they all required GPU usage to be sufficiently quick.

In the context of Neuphonic - building a low-latency, lightweight TTS models - I thought it would be much more fitting to aim for a lightweight chunking tool first (and discuss extensions later).

Hence, `chunk_tts.py` contains a set of heuristics that are lightweight and runs quickly.

### 3.2. The text preprocessing (redundancy removal + normalization)

**Remove any characters or elements that are redundant or unnecessary for speech generation.**

#### Removing redundant dashes

We detect patterns like `-some words-` using regex, meaning it's an "aside" point. These are typically prefixed by a discourse word (e.g. `-particularly ...`). See `DISCOURSE_RHS` in the code for a list of common discourse words.

e.g. `...-though possibly inaccurately-...` transforms to `..., though possibly inaccurately, ...`

This preserves meaning but expresses it in a way TTS handles more naturally (commas instead of weird dash pauses).

We avoid converting true hyphenated compounds like `sweet-tart` which actually make sense. If they don't match the discourse pattern, we leave them as-is.

#### Removing line breaks

We remove single newline characters because they are redundant and seem noisy. We keep double newlines because they indicate paragraphs, which are structurally helpful.

### 3.3. The chunking heuristics

- Split into paragraphs using blank lines. Paragraphs are hard boundaries (no chunk crosses them).

- Within each paragraph, we:
  - Split into sentences using major punctuations (`.`, `?`, `!`) as sentence ending characters.
  - If a sentence is ≤ 200 characters, we emit it as a chunk with rule: "sentence".
  - If a sentence is > 200 characters, we split using minor punctuations (`,`, `;`)
  - If punctuation rules fail to chunk below 200 characters, we break into clauses at subordinators (e.g. `when`, `where`, `while`, etc). See `SUBORDINATORS` in the code.
  - Finally, if all else fails, we break at the length limit (200).

Result: chunks are usually whole sentences; long sentences become clause-sized pieces that sound natural for TTS and stay under ~200 characters.

#### Reasoning (Why these choices help TTS)

We try to keep each chunk as one coherent thought, with punctuation that matches how you'd speak it aloud.

- Paragraphs are structural units, we don't cross them.
- Sentences are natural prosodic units, simple default chunks.
- Commas are softer boundaries, another simple default.
- Subordinators (when/while/before/until/...) mark clause boundaries, which are good secondary pause points for long sentences.


## 4. Extending to Other Languages

Some of the things we would need to change to support languages beyond English:

- Sentence boundaries: different punctuation (e.g. `。` in Japanese/Chinese, `¿/¡` in Spanish).
- Clause markers: other languages have different subordinators or conjunctions; our English list won't apply.
- Dash conventions: usage and meaning can differ by language.
- Tokenization: some languages don't use spaces (Chinese, Japanese, Thai), so breaking the text at a space won't work. (We would need to be careful not to break characters that go together)
- Prosody: average sentence lengths and natural pause points vary across languages.

To build a multi-lingual tool for text chunking, we would need a robust heuristics engine that runs on per-language custom configs (sentence-ending marks, subordinator lists, discourse rules, language-aware tokenizer). 

Crucially, we would need domain experts in natural languages to craft such heuristics configs.

## 5. Thinking Far Ahead

There are improvements to be made on these simple heuristics:

- Short chunks can be merged: if chunks are semantically relevant, they can be merged while staying under the 200 character limit. This would give the TTS model more context for natural speech.
- Semantics, semantics: the current version uses simple heuristic. ML models can more accurately capture these complex rules (especially neural nets) across languages. That may help produce better heuristics.
  - Making ML-based approaches practical: ML-based methods may be sufficiently performant if we can shrink the neural net size required to capture their intricacies. (so they can run on device!). We start with heuristics and supervised learning to teach them, then find ways to shrink the models.
- Streaming: currently the script takes a whole text input and chunks them. A more TTS-friendly version would consume from a stream and generate chunks on the fly.
