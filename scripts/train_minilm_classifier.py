#!/usr/bin/env python3
"""Bootstrap a MiniLM multi-head routing artifact from labeled JSONL data.

This is a lightweight starter scaffold for issue #99. It does not fine-tune
MiniLM itself; instead it encodes each training example with
sentence-transformers/all-MiniLM-L6-v2 and exports one centroid-based linear
head per routing dimension for the Rust runtime.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


HEADS = ("task_type", "complexity", "persona", "domain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL dataset path")
    parser.add_argument("--output", required=True, help="Artifact JSON path")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in ("text", *HEADS) if not row.get(field)]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing required field(s): {', '.join(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no training rows")
    return rows


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def l2_normalize(vector):
    import numpy as np

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def build_head(rows: list[dict], embeddings, field: str) -> dict:
    import numpy as np

    buckets: dict[str, list] = defaultdict(list)
    for row, embedding in zip(rows, embeddings):
        buckets[normalize_label(row[field])].append(embedding)

    labels = sorted(buckets)
    weights = []
    for label in labels:
        centroid = np.mean(np.stack(buckets[label], axis=0), axis=0)
        weights.append(l2_normalize(centroid).tolist())

    return {
        "labels": labels,
        "weights": weights,
        "bias": [0.0 for _ in labels],
    }


def describe_label_counts(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {field: defaultdict(int) for field in HEADS}
    for row in rows:
        for field in HEADS:
            counts[field][normalize_label(row[field])] += 1
    return {field: dict(sorted(values.items())) for field, values in counts.items()}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "This scaffold requires `numpy` and `sentence-transformers`.\n"
            "Install them with:\n"
            "  pip install numpy sentence-transformers"
        ) from exc

    rows = load_rows(input_path)
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        [row["text"] for row in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    artifact = {
        "version": 1,
        "embedding_dim": int(embeddings.shape[1]),
    }
    for field in HEADS:
        artifact[field] = build_head(rows, embeddings, field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    summary = {
        "examples": len(rows),
        "embedding_dim": artifact["embedding_dim"],
        "labels": describe_label_counts(rows),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
