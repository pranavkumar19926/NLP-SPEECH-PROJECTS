# Cross-Lingual & Cross-Modal Representation Analysis in SONAR

**Analysis of Cross-Lingual and Cross-Modal Representations in SONAR: A Multimodal Foundation Model for Text and Speech**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-Open-orange.svg)
![Method](https://img.shields.io/badge/Method-SVCCA-green.svg)


> An intrinsic, model-agnostic study of how multimodal foundation models — **SONAR**, **SeamlessM4T**, and **SALMONN** — internally encode the *same* semantic content across **30 languages** and across the **speech** and **text** modalities, measured with **Singular Vector Canonical Correlation Analysis (SVCCA)** on the **FLEURS** dataset.

---

## Table of Contents

- [Overview](#overview)
- [Motivation & Research Questions](#motivation--research-questions)
- [Key Findings](#key-findings)
- [Models Analyzed](#models-analyzed)
- [Method](#method)
- [Experimental Pipeline](#experimental-pipeline)
- [Results](#results)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Challenges & Limitations](#challenges--limitations)
- [Future Work](#future-work)
- [References](#references)
- [Authors](#authors)

---

## Overview

Modern foundation models aim to learn representations that are **agnostic to surface-level differences** — whether those differences come from the *language* of the input or its *modality* (spoken vs. written). This project probes whether that goal is actually achieved inside the network.

We focus on **SONAR** (Sentence-level Multimodal and Language-Agnostic Representations, Meta AI Research), which produces fixed-size sentence embeddings aligned across 200+ languages and both modalities via a multilingual text encoder (initialized from NLLB), monolingual speech encoders (Wav2Vec2-BERT 2.0), a pooling mechanism, and an **MSE-based alignment loss**. We compare it against two architecturally distinct models — the encoder–decoder **SeamlessM4T** and the decoder-only LLM **SALMONN** — to isolate what role the *training objective* plays versus the *architecture*.

Because the modalities have different feature dimensions, standard similarity-retrieval tools do not apply. We instead use **SVCCA**, which is invariant to affine transformations and therefore well-suited for comparing representations across modalities and architectures.

---

## Motivation & Research Questions

1. **Depth** — How do cross-modal (speech vs. text) representations evolve across the layers of the model?
2. **Resource level** — How effectively does each model bridge the gap between high-, medium-, and low-resource languages?
3. **Gap comparison** — How does the **modality gap** compare to the **language gap** within the same representation space?

---

## Key Findings

| # | Finding | Takeaway |
|---|---------|----------|
| 1 | **Cross-modal representations converge with depth.** | Similarity follows a characteristic **V-shape** — high at the input embeddings (~0.90), dipping at early layers (~0.81 @ layer 4) as surface features are processed, then climbing to ~0.95 by the final layer. |
| 2 | **Length adaptation matters — but mostly for high-resource languages.** | SeamlessM4T's M-Adapter and SALMONN's Q-Former give small (or even negative) gains. Only **SONAR's pooling + MSE alignment** delivers a large gain that *also* helps low-resource languages. |
| 3 | **Speech has larger cross-lingual gaps than text.** | Speech carries more nuisance variation (speaker, accent, pace, recording conditions). Final-layer cross-lingual gap ≈ **+0.054** in favor of text. |
| 4 | **Tokenizer coverage is a measurable bottleneck.** | SALMONN inherits LLaMA's 32k vocab, fragmenting diverse scripts. Shared-token proportion correlates with pairwise SVCCA similarity (**r = 0.228, p = 1.48×10⁻⁶**). |
| 5 | **Modality gap > language gap — except in SONAR.** | For SeamlessM4T and SALMONN, a sentence's *translation in another language* can be more similar than its *own audio*. SONAR's explicit alignment training **inverts** this ordering. |

> **Practical recommendation:** For low-resource or zero-shot speech–text systems, initialize from models explicitly trained for modality- and language-alignment (like SONAR) rather than from larger but unaligned general-purpose multimodal LLMs.

---

## Models Analyzed

| Aspect | SeamlessM4T | SONAR | SALMONN |
|--------|-------------|-------|---------|
| Architecture | Encoder–decoder | Sentence embedding | Decoder-only LLM |
| Text backbone | NLLB | NLLB | Vicuna-7B (Llama-2) |
| Speech backbone | W2v-BERT 2.0 | Wav2Vec2-BERT 2.0 | Whisper + BEATs |
| # Layers | 24 (encoder) | 24 (encoder) | 32 (decoder) |
| Feature dim | 1024 | 1024 | 2048 / 4096 |
| Length adaptation | M-Adapter | Mean / attn-pool | Window Q-Former |
| Explicit alignment loss | — | **MSE** | — |

---

## Method

### SVCCA (Singular Vector Canonical Correlation Analysis)

Given two activation matrices **X ∈ ℝ^(Fx×M)** and **Y ∈ ℝ^(Fy×N)** (with M = N), SVCCA:

1. Applies **SVD** and retains the top singular vectors explaining **90% of variance**.
2. Runs **CCA** on the reduced vectors.
3. Reports the **mean correlation coefficient** as the similarity score in `[0, 1]`.

A numerical stability constant `ε = 1e-10` is used throughout.

### Three Similarity Comparisons

- **Intra-lingual cross-modal:** `text(sentence, L₁)` vs. `speech(sentence, L₁)`
- **Cross-lingual text:** `text(sentence, L₁)` vs. `text(sentence, L₂)`
- **Cross-lingual speech:** `speech(sentence, L₁)` vs. `speech(sentence, L₂)`

### Dataset — FLEURS

The **FLEURS** test split (an n-way parallel speech dataset paired with FLoRes-101 transcripts) is used. **30 languages** were selected from the 102 available to balance:

- **Scripts:** Latin, Cyrillic, Devanagari, Arabic, Han, Japanese, Korean, Ethiopic, Hebrew, Khmer, Lao, Tamil, Telugu, Malayalam, Georgian, Armenian, Thai, Bengali.
- **Families:** Indo-European, Afro-Asiatic, Sino-Tibetan, Dravidian, Uralic, Austronesian, Austroasiatic, Tai-Kadai, Atlantic-Congo, Turkic, Koreanic, Japonic, Kartvelian.
- **Resource levels:** high, medium, and low.

---

## Experimental Pipeline

```text
1. Load FLEURS test split (30 languages)
2. Deduplicate audio and normalize transcripts
3. Load model (SeamlessM4T / SONAR / SALMONN)
4. Forward pass → extract per-layer activations
5. Mean-pool over the sequence-length dimension
6. Compute SVCCA pairwise (cross-modal & cross-lingual)
7. Aggregate by resource level → visualize
```

---

## Results

### Cross-modal similarity by resource level (SeamlessM4T)

| Resource Level | Layer 4 | Layer 12 | Final Layer | After Length-Adapt. | Random Baseline |
|----------------|:-------:|:--------:|:-----------:|:-------------------:|:---------------:|
| High-resource | 0.85 | 0.89 | 0.915 | 0.93 | 0.9210 |
| Medium-resource | 0.83 | 0.87 | 0.900 | 0.91 | 0.9210 |
| Low-resource | 0.82 | 0.85 | 0.880 | 0.88 | 0.9210 |

### Length-adaptation gain (Δ cross-modal SVCCA)

| Model | High | Medium | Low | Note |
|-------|:----:|:------:|:---:|------|
| SeamlessM4T | +0.015 | +0.010 | +0.001 | M-Adapter, slight gain |
| **SONAR** | **+0.033** | **+0.039** | **+0.050** | Pooling + MSE, large gain |
| SALMONN | −0.006 | −0.002 | −0.001 | Q-Former, minimal |

### Cross-lingual: text vs. speech (SeamlessM4T, final layers)

| Layer | Text | Speech | Gap |
|:-----:|:----:|:------:|:---:|
| 0 | 0.892 | 0.706 | +0.186 |
| 8 | 0.913 | 0.676 | +0.237 |
| 16 | 0.943 | 0.773 | +0.170 |
| 24 (final) | 0.956 | 0.902 | +0.054 |

### Summary performance metrics

| Metric | SeamlessM4T | SONAR | SALMONN |
|--------|:-----------:|:-----:|:-------:|
| Cross-modal sim. (final layer) | ~0.91 | **~0.94** | ~0.88 |
| Cross-lingual text sim. | ~0.92 | **~0.95** | ~0.89 |
| Cross-lingual speech sim. | ~0.88 | **~0.93** | ~0.86 |
| Modality gap reduced? | Partially | **Strongly** | Partially |
| Language gap reduced? | Strongly | Strongly | Weakly (early layers) |

---

## Getting Started

The full implementation lives in a single, reproducible Google Colab notebook.

▶️ **[Open the Colab Notebook](https://colab.research.google.com/drive/1xXpVw-fpI36CcCfE9G19c8XnDerElWij?usp=sharing)**

### Run on Colab (recommended)

1. Open the notebook via the link above.
2. Set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`).
3. Run all cells. Representation extraction completes in ~2.5 minutes; each analysis section takes a few seconds.

### Run locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets numpy scipy scikit-learn matplotlib
```

> **Experimental setup (reference run):** NVIDIA Tesla T4 (15.6 GB VRAM), PyTorch 2.x + CUDA, fp16 inference. Primary model `facebook/seamless-m4t-v2-large` (~2.3B params). Layers analyzed: `[0, 4, 8, 12, 16, 20, 24]`. SVCCA at 90% variance, `ε = 1e-10`. Random baseline ≈ 0.9210.

---

## Repository Structure

```text
.
├── README.md                      # You are here
├── notebooks/
│   └── sonar_analysis.ipynb       # Main Colab notebook
├── src/
│   ├── svcca.py                   # SVCCA similarity implementation
│   ├── data.py                    # FLEURS loading & preprocessing
│   ├── extract.py                 # Per-layer activation extraction
│   └── analyze.py                 # Cross-modal / cross-lingual analysis
├── figures/                       # Line plots, bar charts, t-SNE projections
└── report/
    └── Project_Report_SONAR_Analysis.pdf
```

> Adjust this layout to match how you organize the repo. If everything stays in the notebook, you can keep just `notebooks/` and `report/`.

---

## Challenges & Limitations

- **Compute constraints:** SALMONN-7B needs ~14 GB in fp16 — worked around with bfloat16 inference, one-language-at-a-time processing, and persisting activations as `.npy` shards.
- **Long inference times:** A full pass over 30 languages (~250 sentences each) across three models took tens of GPU-hours; Colab timeouts required frequent checkpointing.
- **Language coverage mismatch:** SONAR does not support 6 of the 30 languages (Khmer, Lao, Georgian, Armenian, Amharic, Shona); same-resource-level substitutes were used.
- **SVCCA cost:** SALMONN's 4096-dim activations made SVD memory-hungry and numerically delicate.
- **Variable sequence lengths:** Speech utterances are 50–100× longer than transcripts; mean-pooling loses fine-grained temporal information.
- **Domain bias:** FLEURS derives from FLoRes-101 (Wikipedia) — formal, clean text unlike spontaneous conversational speech.

---

## Future Work

- Re-run on newer models (**Qwen2-Audio**, **Llama 3**) to test whether gaps persist at scale.
- Extend to the **image** modality (BLIP-2, LLaVA).
- Add **logit-lens** probing for decoder-only models.
- Pair SVCCA scores with **downstream metrics** (ASR WER, speech-translation BLEU).
- Scale from 30 → all **102 FLEURS languages**, including conversational/code-switched speech.
- Redesign SALMONN's tokenizer with better low-resource script coverage.

---

## References

Selected key references (see the [full report](report/Project_Report_SONAR_Analysis.pdf) for the complete list):

1. Lee et al., *How do multimodal foundation models encode text and speech?*, NAACL-HLT 2025.
2. Seamless Communication et al., *SeamlessM4T*, arXiv:2308.11596, 2023.
3. Duquenne, Schwenk & Sagot, *SONAR: sentence-level multimodal and language-agnostic representations*, arXiv:2308.11466, 2023.
4. Tang et al., *SALMONN: towards generic hearing abilities for large language models*, ICLR 2024.
5. Raghu et al., *SVCCA: singular vector canonical correlation analysis for deep learning dynamics*, NeurIPS 2017.
6. Conneau et al., *FLEURS: few-shot learning evaluation of universal representations of speech*, IEEE SLT 2023.

---

## Authors

Developed as a project in the **Department of Computer Science and Engineering**, **Thapar Institute of Engineering and Technology**.

- **Gorthi Sai Madhuri** — 102317063
- **Devulapally Pranav Kumar** — 102317278

**Guides:** Dr. Simran · Mr. Jasmeet Singh

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sonar_repr_analysis_2025,
  title        = {Analysis of Cross-Lingual and Cross-Modal Representations in SONAR:
                  A Multimodal Foundation Model for Text and Speech},
  author       = {Gorthi, Sai Madhuri and Devulapally, Pranav Kumar},
  year         = {2025},
  institution  = {Thapar Institute of Engineering and Technology},
  note         = {Department of Computer Science and Engineering}
}
```
