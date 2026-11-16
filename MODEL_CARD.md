---
language:
- zh
license: apache-2.0
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
library_name: qwen_tts
pipeline_tag: text-to-speech
tags:
- tts
- chinese
- single-speaker
- qwen3-tts
---

# Model card

## Overview

Beijing Young Men TTS is a single-speaker supervised fine-tune of Qwen3-TTS-12Hz-1.7B-Base, released by the voice owner. It generates a fixed personal Mandarin male voice from text through `generate_custom_voice`.

This is a full custom-voice inference checkpoint, not an adapter. The released checkpoint uses `speaker="user_speaker"` and does not require an additional reference recording.

## Provenance

- Base: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- Base revision: `fd4b254389122332181a7c3db7f27e918eec64e3`.
- Training date: 2026-10-24.
- Split: 122 training clips, 22 validation clips, 16 held-out test clips.
- Training: 3 epochs of single-speaker SFT. Selected checkpoint: epoch 3.
- Learning rate: `2e-6`. Batch size: `2`. Gradient accumulation: `4`.
- Inference weights: BF16.
- The learned fixed-speaker vector is stored in codec embedding row 3000. The export omits the speaker encoder, which is not used by fixed-speaker custom-voice generation.
- The checkpoint includes its speech tokenizer and required text tokenizer files.

The source recordings were supplied by the voice owner. Training data and original recordings are not distributed in this release.

## Files and integrity

The main weight is `3,833,402,520` bytes, SHA-256:
`c03fa20db0b545e2546bca808753c9ce40dbfc5f7435cea8f81d7e13e2867678`.

Speech tokenizer weight is `682,293,092` bytes, SHA-256:
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.

Release assets split the main weight to satisfy GitHub's per-asset limit. The downloader reconstructs the exact original file and verifies both parts and final files. See `model-manifest.json`.

## Intended use and behavior

Mandarin narration, reading and technical explanations using the released fixed voice. The repository name does not establish evaluated Beijing-dialect capability: the checkpoint configuration marks this speaker as not dialect-specific.

Other base-model language identifiers may be accepted by the implementation, but this release is not a multilingual quality benchmark. Use `Chinese` for the documented workflow.

Generation can vary with text, punctuation and sampling. Long texts should be split at natural boundaries. Listen for omitted or repeated words, truncated endings, numerical pronunciation and sentence transitions.

## Validation

The original local validation used Python 3.12, qwen-tts 0.1.1, PyTorch / torchaudio 2.8.0, transformers 4.57.3 and accelerate 1.12.0. CPU and a split GPU/CPU path were used for short and narration samples. GPU generation used BF16 talker weights and a CPU FP32 speech decoder.

The published examples are previously generated outputs of this checkpoint, including a 9.92-second segment from a new-text narration test and a 1.60-second short sentence. They are not raw training recordings.

No broad naturalness score, identity similarity benchmark, Beijing-dialect score or hardware-independent speed guarantee is claimed.

## License

Apache-2.0. The upstream Qwen3-TTS model and implementation retain their original attribution. See LICENSE and NOTICE.
