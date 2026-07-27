# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-07-26

First release of `mlx-nanbeige` as a standalone package. 0.1.0 lived inside the
evaluation harness repo and was never published.

### Added

- **`py.typed`** — the package's type hints are now usable downstream.
- **`pull(quant)`** — `snapshot_download` wrapper with the published quant repo
  ids baked in, so `mlx_lm.load(pull("4bit"))` needs no manual path handling.
- **`upload.py`** — publishes a converted quant to the Hub with a correct model
  card: Apache-2.0 frontmatter, `base_model`, upstream `LICENSE` copied and a
  `NOTICE` stating the modification (§4 compliance). `--dry-run` renders the card
  and lists files without touching the network.
- **Prefill-vs-incremental-decode consistency test.** The looped architecture
  needs `num_loops × num_hidden_layers` = 44 KV slots that must advance in
  lockstep; the test asserts one-shot prefill logits match incremental decode,
  and is mutation-verified to fail if the loop virtualisation is removed.

### Changed

- **Split out of the eval repo.** `model.py` is now the single source of truth
  and is copied verbatim into each converted repo as the `model_file`. The three
  copies inside `models/nanbeige-mlx-*bit/` are generated artifacts.
- **`prepare_source` no longer mutates the source directory.** It stages a copy
  in a tempdir and symlinks the safetensors shards rather than rewriting the
  caller's `config.json` and dropping a file into it. Pointing `--src` at an HF
  cache snapshot no longer corrupts it.
- **Tokenizer files survive conversion.** `added_tokens.json`,
  `special_tokens_map.json`, `tokenizer.model` and `chat_template.jinja` are
  carried through; transient keys (`is_local`, `local_files_only`, `backend`)
  that leaked from `tokenizer.init_kwargs` are stripped, and
  `model_max_length` is set to the real 262 144. Conversion now asserts
  `AutoTokenizer.from_pretrained(out)` round-trips `<|im_end|>` → 166101 — a
  conversion that silently loses EOS produces a model that never stops.
- **Dependency bounds are upper-bounded** (`mlx-lm>=0.31,<0.33`,
  `mlx>=0.32,<0.34`). `model.py` imports mlx-lm internals and is frozen into
  every published weight repo, where it cannot be patched for users who already
  downloaded it. Relaxing the bound is a decision to make after testing.

### Fixed

- **`o_proj` honours `attention_bias`** instead of hardcoding `bias=False`.
  Inert for this checkpoint (`attention_bias: false`) but wrong for any other.
- **`make_cache()` documents the KV ceiling.** `--max-kv-size` is inert whenever
  a model supplies `make_cache` (`make_prompt_cache` skips it), and the full
  262 144 context needs ~47 GB of KV across 44 slots. `--kv-bits` still works.

### Notes on correctness

Verified: per-layer arithmetic agrees with the checkpoint's own
`NanbeigeDecoderLayer` to fp32 precision; the 44-slot cache passes the
prefill-vs-decode test; the port's two code paths are bit-identical.

Open: end-to-end next-token logit cosine against the HF reference is 0.847
(top-1 agreement 83%), lower than a faithful port should give. Six candidate
causes have been eliminated by measurement. The full record, including every
falsified hypothesis, is in the eval repo's
[`docs/investigation-log.md`](https://github.com/jishnuvenugopal/nanbeige-mlx-eval/blob/main/docs/investigation-log.md).
Behaviour is unaffected — 26–28/30 on the bilingual agentic suite at every
quantisation level.

## [0.1.0] — unreleased

Initial port, developed inside the evaluation harness repo. Never published.
