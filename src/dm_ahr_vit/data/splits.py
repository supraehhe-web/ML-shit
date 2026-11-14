"""Lesion-grouped splits: union of lesion_id and confirmed near-duplicates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CLASSES


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(c0: pd.DataFrame, near: pd.DataFrame) -> pd.DataFrame:
    """Merge images sharing a lesion_id or a confirmed near-duplicate link."""
    uf = UnionFind()
    for lesion in c0["lesion_id"].unique():
        members = c0.loc[c0["lesion_id"] == lesion, "image_id"].tolist()
        for m in members[1:]:
            uf.union(members[0], m)
    if len(near):
        for r in near.itertuples(index=False):
            uf.union(r.id_a, r.id_b)
    df = c0.copy()
    first_seen: dict = {}
    group_ids = []
    for iid in df["image_id"]:
        root = uf.find(iid)
        if root not in first_seen:
            first_seen[root] = f"G{len(first_seen):05d}"
        group_ids.append(first_seen[root])
    df["group_id"] = group_ids
    return df


def freeze_splits(grouped: pd.DataFrame, seed: int, fractions: dict,
                  out_dir, protocol_hash: str) -> dict:
    """Stratified group split, frozen to CSV plus its SHA-256."""
    rng = np.random.default_rng(seed)
    order = ["train", "val", "test"]
    fr = [fractions[s] for s in order]
    assign: dict = {}
    for dx in CLASSES:
        groups = sorted(grouped.loc[grouped["dx"] == dx, "group_id"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_tr = int(round(fr[0] * n))
        n_va = int(round(fr[1] * n))
        for g in groups[:n_tr]:
            assign[g] = "train"
        for g in groups[n_tr:n_tr + n_va]:
            assign[g] = "val"
        for g in groups[n_tr + n_va:]:
            assign[g] = "test"
    df = grouped.copy()
    df["split"] = df["group_id"].map(assign)
    df = df[["image_id", "lesion_id", "group_id", "dx", "split", "sha256"]]
    df = df.sort_values("image_id").reset_index(drop=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "splits.csv", index=False)
    digest = hashlib.sha256((out_dir / "splits.csv").read_bytes()).hexdigest()
    (out_dir / "splits.sha256").write_text(digest + "\n")
    counts = {s: df[df["split"] == s]["dx"].value_counts().to_dict() for s in order}
    return {
        "counts": counts,
        "n_groups": int(df["group_id"].nunique()),
        "splits_sha256": digest,
        "protocol_hash": protocol_hash,
    }


def verify_splits(df: pd.DataFrame) -> dict:
    """No group may appear in more than one split."""
    problems = []
    splits = ["train", "val", "test"]
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            ga = set(df.loc[df["split"] == a, "group_id"])
            gb = set(df.loc[df["split"] == b, "group_id"])
            if ga & gb:
                problems.append(f"group overlap between {a} and {b}: {len(ga & gb)} groups")
    if df["image_id"].duplicated().any():
        problems.append("duplicate image_id in split frame")
    sizes = {s: int((df["split"] == s).sum()) for s in splits}
    return {"ok": not problems, "problems": problems, "sizes": sizes}
