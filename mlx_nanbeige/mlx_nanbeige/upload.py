"""Publish a converted Nanbeige MLX repo to the HuggingFace Hub.

Nanbeige4.2-3B is Apache-2.0, which permits redistribution of derivative works
(quantization is a modification), so §4(d) applies: retain the upstream
LICENSE, retain notices, and *state that you changed the files*. This module
writes proper model-card frontmatter, copies the upstream LICENSE through, and
emits a NOTICE describing the modification, then (unless ``--dry-run``) uploads.

Run with ``--dry-run`` first: it renders the card and lists the files that would
be uploaded without touching the network. The actual upload needs
``huggingface-cli login`` and is the one outward-facing step — by default this
module does NOT push, it prepares everything up to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Apache-2.0 §4(d): state what changed.
NOTICE_TEMPLATE = """\
Nanbeige4.2-3B MLX quantization
================================

This repository contains a derivative of `{base_model}` (Apache-2.0).

Modifications by Jishnu Venugopal ({year}):
  * Converted from the original PyTorch checkpoint to Apple's MLX format.
  * Quantized to {bits}-bit (group size {group_size}) for Apple Silicon.

The original model and its Apache-2.0 license are retained (see LICENSE).
Upstream notices and the full license text are unchanged. This derivative is
redistributed under the same Apache-2.0 license.

The MLX model definition (`nanbeige.py`) is MIT-licensed source code from the
`mlx-nanbeige` project; the weights themselves remain Apache-2.0.
"""


def _card_frontmatter(bits: int, group_size: int) -> str:
    # Proper frontmatter for a bilingual quantized weights repo (C7).
    return (
        "---\n"
        "license: apache-2.0\n"
        "base_model: Nanbeige/Nanbeige4.2-3B\n"
        "language: [en, zh]\n"
        "library_name: mlx\n"
        "pipeline_tag: text-generation\n"
        f"tags: [mlx, nanbeige, looped-transformer, apple-silicon, {bits}-bit]\n"
        f"quantized_from: Nanbeige/Nanbeige4.2-3B\n"
        f"quantization:\n  bits: {bits}\n  group_size: {group_size}\n"
        "---\n"
    )


def _card_body(bits: int, group_size: int) -> str:
    return f"""
# Nanbeige4.2-3B MLX ({bits}-bit)

An MLX conversion and {bits}-bit quantization (group size {group_size}) of
[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) — a
3-billion-parameter **Looped Transformer** (`num_loops=2`, weight-shared over 22
layers for an effective depth of 44). Produced by the independent
[`mlx-nanbeige`](https://github.com/jishnuvenugopal/mlx-nanbeige) project;
not affiliated with the Nanbeige team.

## Load (no conversion step)

```python
from mlx_nanbeige import pull
import mlx_lm
model, tok = mlx_lm.load(pull("{bits}bit"))
```

Weights arrive on first use and are cached in `~/.cache/huggingface`.

## KV-cache ceiling (a real cost of the looped design)

The looped architecture needs `num_loops * num_hidden_layers = 44` KV slots.
At full context that is 44 x 8 KV-heads x 128 dim x 2 (K+V) x 262144 positions
x 2 bytes ~= **47 GB** — unreachable on a 16 GB machine. Because this model
supplies `make_cache`, mlx-lm's `--max-kv-size` knob is inert; use `--kv-bits`
to reduce KV precision instead.

## License

Apache-2.0 (the upstream license; quantization is a modification under §4(d)).
The `nanbeige.py` model definition is MIT-licensed source from `mlx-nanbeige`.
See `LICENSE` and `NOTICE`.
"""


def write_card(model_dir: str | Path, bits: int, group_size: int, *, base_model: str = "Nanbeige/Nanbeige4.2-3B") -> Path:
    """Write README.md, LICENSE (copied from upstream), and NOTICE into ``model_dir``."""
    import datetime
    import shutil

    d = Path(model_dir)
    card = d / "README.md"
    card.write_text(_card_frontmatter(bits, group_size) + _card_body(bits, group_size), encoding="utf-8")

    notice = d / "NOTICE"
    notice.write_text(
        NOTICE_TEMPLATE.format(
            base_model=base_model,
            bits=bits,
            group_size=group_size,
            year=datetime.datetime.now().year,
        ),
        encoding="utf-8",
    )

    # Copy the upstream Apache-2.0 LICENSE through if it's not already present.
    license_path = d / "LICENSE"
    if not license_path.exists():
        # The convert step doesn't carry LICENSE; the publisher is expected to
        # have the upstream checkout (or its LICENSE) available. If we can't find
        # one, write a pointer rather than silently shipping an unlicensed repo.
        candidates = [
            d.parent / "nanbeige42-hf" / "LICENSE",
            d.parent / "LICENSE.apache",
        ]
        src_license = next((c for c in candidates if c.exists()), None)
        if src_license is not None:
            shutil.copy2(src_license, license_path)
        else:
            license_path.write_text(
                "Apache License 2.0 — see https://www.apache.org/licenses/LICENSE-2.0\n"
                "Upstream: https://huggingface.co/Nanbeige/Nanbeige4.2-3B\n",
                encoding="utf-8",
            )
    return card


def upload(model_dir: str | Path, repo_id: str, *, dry_run: bool = True) -> None:
    """Write the card/LICENSE/NOTICE and (unless ``dry_run``) upload to ``repo_id``."""
    d = Path(model_dir)
    cfg = __import__("json").loads((d / "config.json").read_text(encoding="utf-8"))
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = int(q.get("bits", 0))
    group_size = int(q.get("group_size", 64))

    write_card(d, bits, group_size)

    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print(f"repo: {repo_id}")
    print(f"model_dir: {d}")
    print(f"quantization: {bits}-bit, group_size={group_size}")
    print("files:")
    for name in files:
        print(f"  - {name}")

    if dry_run:
        print("\n[dry-run] no upload performed. Re-run without --dry-run to publish.")
        return

    from huggingface_hub import HfApi  # type: ignore

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(folder_id=str(d), repo_id=repo_id, repo_type="model")
    print(f"\nuploaded to {repo_id}")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(
        prog="mlx-nanbeige-upload",
        description="Write a proper model card + LICENSE + NOTICE and upload a Nanbeige MLX quant.",
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="render the card and list files without uploading (default behavior)",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="actually upload (requires `huggingface-cli login`)",
    )
    a = ap.parse_args(argv)
    upload(a.model_dir, a.repo_id, dry_run=not a.yes)


if __name__ == "__main__":  # pragma: no cover
    main()
