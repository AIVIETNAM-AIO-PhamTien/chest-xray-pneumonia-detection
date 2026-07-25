from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _validate_fraction(name: str, value: float, allow_zero: bool = False) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        operator = "0 <=" if allow_zero else "0 <"
        raise ValueError(f"{name} must satisfy {operator} {name} < 1.")


def _normalise_groups(df: pd.DataFrame, group_col: str) -> pd.Series:
    groups = df[group_col].astype("string")
    missing = groups.isna() | groups.str.strip().eq("")
    groups = groups.astype(object)
    for idx in df.index[missing]:
        groups.loc[idx] = f"__row_{idx}"
    return groups.astype(str)


def _group_stratified_split(
    df: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _normalise_groups(df, group_col)
    unique_groups = groups.nunique()
    desired_splits = max(2, round(1.0 / holdout_fraction))
    n_splits = min(desired_splits, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough groups for a group-aware split.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    overall_distribution = df["label"].value_counts(normalize=True).sort_index()
    best = None
    best_score = float("inf")

    for train_idx, holdout_idx in splitter.split(df, df["label"], groups):
        train_part = df.iloc[train_idx]
        holdout_part = df.iloc[holdout_idx]
        size_error = abs(len(holdout_part) / len(df) - holdout_fraction)
        holdout_distribution = (
            holdout_part["label"].value_counts(normalize=True).sort_index()
        )
        distribution_error = (
            overall_distribution.subtract(holdout_distribution, fill_value=0.0)
            .abs()
            .sum()
        )
        score = size_error + distribution_error
        if score < best_score:
            best_score = score
            best = (train_part, holdout_part)

    if best is None:
        raise RuntimeError("Could not construct a group-aware split.")
    return best[0].copy(), best[1].copy()


def _stratified_split(
    df: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    group_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_fraction("holdout_fraction", holdout_fraction)
    if df["label"].nunique() < 2:
        raise ValueError("At least two classes are required for stratification.")

    if group_col and group_col in df.columns:
        try:
            return _group_stratified_split(
                df=df,
                holdout_fraction=holdout_fraction,
                seed=seed,
                group_col=group_col,
            )
        except (ValueError, RuntimeError) as exc:
            warnings.warn(
                f"Group-aware split failed ({exc}); falling back to image-level "
                "stratification.",
                RuntimeWarning,
            )

    train_part, holdout_part = train_test_split(
        df,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=df["label"],
    )
    return train_part.copy(), holdout_part.copy()


def _assign_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    result = df.copy()
    result["split"] = name
    return result


def _check_hash_leakage(df: pd.DataFrame) -> None:
    if "sha256" not in df.columns:
        return

    split_counts = df.groupby("sha256")["split"].nunique()
    leaking_hashes = split_counts[split_counts > 1]
    if not leaking_hashes.empty:
        examples = leaking_hashes.index[:5].tolist()
        raise ValueError(
            f"Duplicate image content crosses splits for {len(leaking_hashes)} "
            f"hashes. Examples: {examples}"
        )


def _check_group_leakage(df: pd.DataFrame, group_col: Optional[str]) -> None:
    if not group_col or group_col not in df.columns:
        return

    groups = _normalise_groups(df, group_col)
    check = pd.DataFrame({"group": groups, "split": df["split"]})
    split_counts = check.groupby("group")["split"].nunique()
    leaking_groups = split_counts[split_counts > 1]
    if not leaking_groups.empty:
        examples = leaking_groups.index[:5].tolist()
        raise ValueError(
            f"{len(leaking_groups)} groups from '{group_col}' cross final splits. "
            f"Examples: {examples}"
        )


def create_splits(
    manifest: pd.DataFrame,
    strategy: str = "official",
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    group_col: Optional[str] = "patient_id",
) -> pd.DataFrame:
    """Create train/val/test assignments.

    ``official`` keeps the supplied test folder untouched and creates a new
    validation split from the original train+val pool.

    ``paper`` merges original train+val and makes a new stratified 80/20-style
    test split. Set ``val_size=0`` to match a strict train/test-only setup.

    ``all`` ignores source folders and splits every manifest row.
    """

    required = {"path", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    strategy = strategy.lower()
    if strategy not in {"official", "paper", "all"}:
        raise ValueError("strategy must be official, paper, or all.")
    _validate_fraction("test_size", test_size)
    _validate_fraction("val_size", val_size, allow_zero=True)
    if strategy in {"paper", "all"} and test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    df = manifest.reset_index(drop=True).copy()

    if strategy == "official":
        if "source_split" not in df.columns:
            raise ValueError("official strategy requires a source_split column.")
        test_df = df[df["source_split"].str.lower().eq("test")].copy()
        train_pool = df[df["source_split"].str.lower().isin(["train", "val"])].copy()
        if test_df.empty or train_pool.empty:
            raise ValueError("official strategy requires non-empty train and test folders.")

        if val_size > 0:
            train_df, val_df = _stratified_split(
                train_pool,
                holdout_fraction=val_size,
                seed=seed,
                group_col=group_col,
            )
        else:
            train_df, val_df = train_pool, train_pool.iloc[0:0].copy()

    else:
        if strategy == "paper" and "source_split" in df.columns:
            pool = df[df["source_split"].str.lower().isin(["train", "val"])].copy()
        else:
            pool = df.copy()

        train_val_df, test_df = _stratified_split(
            pool,
            holdout_fraction=test_size,
            seed=seed,
            group_col=group_col,
        )

        if val_size > 0:
            relative_val_size = val_size / (1.0 - test_size)
            train_df, val_df = _stratified_split(
                train_val_df,
                holdout_fraction=relative_val_size,
                seed=seed + 1,
                group_col=group_col,
            )
        else:
            train_df, val_df = train_val_df, train_val_df.iloc[0:0].copy()

    parts = [_assign_split(train_df, "train")]
    if not val_df.empty:
        parts.append(_assign_split(val_df, "val"))
    parts.append(_assign_split(test_df, "test"))

    split_df = pd.concat(parts, ignore_index=True)
    split_df = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if split_df["path"].duplicated().any():
        duplicates = split_df.loc[split_df["path"].duplicated(), "path"].head().tolist()
        raise ValueError(f"Duplicate paths found after splitting: {duplicates}")

    _check_hash_leakage(split_df)
    _check_group_leakage(split_df, group_col)
    return split_df


def print_split_summary(df: pd.DataFrame) -> None:
    print("Counts by final split and class:")
    print(
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print("\nSplit proportions:")
    print((df["split"].value_counts(normalize=True).sort_index() * 100).round(2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CXR data splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=["official", "paper", "all"],
        default="official",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-col",
        default="patient_id",
        help="Set to an empty string to disable group-aware splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.input)
    group_col = args.group_col.strip() or None
    result = create_splits(
        manifest=manifest,
        strategy=args.strategy,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        group_col=group_col,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote: {output}")
    print_split_summary(result)


if __name__ == "__main__":
    main()
