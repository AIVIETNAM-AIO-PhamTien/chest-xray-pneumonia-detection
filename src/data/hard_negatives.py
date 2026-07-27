"""Pick out the normal cases two frozen models already find difficult.

The models agree on most normals and disagree on the ones near the decision
boundary. Those are the cases worth weighting more heavily, and they can be
identified from development predictions alone.

Difficulty is the mean of each model's percentile rank. Ranks make the two
scales comparable without either model's confidence dominating, and here they
are safe in a way they were not for the ensemble: they only order labelled
normals inside one fold's training split, and never become a threshold applied
to a differently balanced sample.

Two boundaries matter and are enforced rather than assumed. Ranks are computed
inside each fold's own training normals, so the cutoff never sees validation or
benchmark data. And difficulty is defined among normals only, since ranking
across both classes would mostly recover the class itself.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

#: Fraction of each fold's training normals treated as hard.
HARD_FRACTION = 0.25


def merge_teacher_predictions(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join several models' out-of-fold group predictions into one table.

    Args:
        frames: Mapping of model name to a frame with group_id, label and
            probability columns.

    Returns:
        One row per group with a probability column per model.

    Raises:
        ValueError: If the models disagree on membership or on labels.
    """
    names = list(frames)
    merged = None
    for name in names:
        frame = frames[name][["group_id", "label", "p_pneumonia"]].copy()
        if frame["group_id"].duplicated().any():
            raise ValueError(f"{name}: group_id lặp lại trong bảng OOF")
        frame = frame.rename(columns={"p_pneumonia": f"p_{name}"})
        if merged is None:
            merged = frame
            continue
        before = len(merged)
        merged = merged.merge(frame, on="group_id", how="inner",
                              suffixes=("", "_other"))
        if len(merged) != before or len(merged) != len(frame):
            raise ValueError(f"{name}: tập group không khớp với mô hình trước")
        if not (merged["label"] == merged["label_other"]).all():
            raise ValueError(f"{name}: nhãn không khớp với mô hình trước")
        merged = merged.drop(columns="label_other")
    return merged.sort_values("group_id").reset_index(drop=True)


def hardness_for_fold(teachers: pd.DataFrame, training_groups,
                      fraction: float = HARD_FRACTION) -> pd.DataFrame:
    """Rank one fold's training normals by how hard the teachers found them.

    Args:
        teachers: Merged prediction table from :func:`merge_teacher_predictions`.
        training_groups: Group ids in this fold's training split.
        fraction: Portion of normals to mark as hard.

    Returns:
        One row per training normal group, with its hardness and flag.

    Raises:
        ValueError: If the fold's training split contains no normal groups.
    """
    wanted = set(training_groups)
    block = teachers[teachers["group_id"].isin(wanted)
                     & (teachers["label"] == 0)].copy()
    if block.empty:
        raise ValueError("Fold không có group NORMAL nào trong training split")

    columns = [c for c in block.columns if c.startswith("p_")]
    for column in columns:
        # Rank inside this fold's training normals only. Ranking over a wider
        # pool would let groups the fold never trains on move the cutoff.
        block[f"rank_{column[2:]}"] = (rankdata(block[column])
                                       / (len(block) + 1.0))
    block["hardness"] = block[[f"rank_{c[2:]}" for c in columns]].mean(axis=1)

    n_hard = int(round(fraction * len(block)))
    cutoff = (block["hardness"].nlargest(n_hard).min() if n_hard
              else np.inf)
    block["hard_normal"] = block["hardness"] >= cutoff
    block.attrs["cutoff"] = float(cutoff) if n_hard else float("nan")
    return block.sort_values("hardness", ascending=False).reset_index(drop=True)


def build_hardness_tables(teachers: pd.DataFrame, folds,
                          fraction: float = HARD_FRACTION):
    """Build a hardness table for every fold.

    Args:
        teachers: Merged prediction table.
        folds: Sequence of manifests, each with group_id and split columns.
        fraction: Portion of normals to mark as hard.

    Returns:
        Tuple of (mapping fold index to table, per-fold summary frame).
    """
    tables, rows = {}, []
    for index, manifest in enumerate(folds):
        training = manifest.loc[manifest["split"] == "train", "group_id"].unique()
        table = hardness_for_fold(teachers, training, fraction)
        tables[index] = table

        hard = table[table["hard_normal"]]
        rest = table[~table["hard_normal"]]
        summary = {"fold": index,
                   "n_train_normal_groups": int(len(table)),
                   "n_hard_normal_groups": int(len(hard)),
                   "hard_fraction": float(hard.shape[0] / len(table)),
                   "hardness_cutoff": table.attrs["cutoff"]}
        for column in (c for c in table.columns if c.startswith("p_")):
            model = column[2:]
            summary[f"median_{model}_hard"] = float(hard[column].median())
            summary[f"median_{model}_other"] = float(rest[column].median())
        rows.append(summary)
    return tables, pd.DataFrame(rows)


def flag_images(manifest: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Spread a group-level hard flag onto every image in that group.

    Args:
        manifest: Image-level rows with a group_id column.
        table: Hardness table for the same fold.

    Returns:
        Boolean array aligned with the manifest rows.
    """
    hard = set(table.loc[table["hard_normal"], "group_id"])
    return manifest["group_id"].isin(hard).to_numpy()
