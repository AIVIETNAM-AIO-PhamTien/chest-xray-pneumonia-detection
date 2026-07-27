"""Tests for hard-negative selection and the weighted objective."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.data.hard_negatives import (
    build_hardness_tables,
    flag_images,
    hardness_for_fold,
    merge_teacher_predictions,
)


@pytest.fixture
def teachers():
    """Two models' out-of-fold group predictions over a small development set.

    Returns:
        The merged prediction table.
    """
    rng = np.random.default_rng(0)
    n_normal, n_pneumonia = 200, 300
    groups = ([f"normal:{i}" for i in range(n_normal)]
              + [f"pneumonia:{i}" for i in range(n_pneumonia)])
    labels = np.r_[np.zeros(n_normal, int), np.ones(n_pneumonia, int)]
    frames = {}
    for offset, name in enumerate(("a", "b")):
        base = rng.beta(1.5, 6.0, n_normal)
        positive = rng.beta(6.0, 1.5, n_pneumonia)
        frames[name] = pd.DataFrame({
            "group_id": groups, "label": labels,
            "p_pneumonia": np.r_[base, positive]})
    return merge_teacher_predictions(frames)


@pytest.fixture
def folds():
    """Five manifests whose training splits cover different groups.

    Returns:
        List of manifest frames.
    """
    rng = np.random.default_rng(1)
    groups = ([f"normal:{i}" for i in range(200)]
              + [f"pneumonia:{i}" for i in range(300)])
    out = []
    for k in range(5):
        assignment = rng.permutation(len(groups))
        split = np.where(assignment < 100, "val", "train")
        out.append(pd.DataFrame({"group_id": groups, "split": split}))
    return out


def test_merge_rejects_duplicate_groups():
    """A repeated group would double-count in the ranking."""
    frame = pd.DataFrame({"group_id": ["a", "a"], "label": [0, 0],
                          "p_pneumonia": [0.1, 0.2]})
    with pytest.raises(ValueError, match="lặp lại"):
        merge_teacher_predictions({"m": frame})


def test_merge_rejects_disagreeing_labels():
    """Two models must describe the same groups with the same labels."""
    a = pd.DataFrame({"group_id": ["x"], "label": [0], "p_pneumonia": [0.1]})
    b = pd.DataFrame({"group_id": ["x"], "label": [1], "p_pneumonia": [0.9]})
    with pytest.raises(ValueError, match="nhãn không khớp"):
        merge_teacher_predictions({"a": a, "b": b})


def test_no_pneumonia_group_is_ever_marked_hard(teachers, folds):
    """Hardness is defined among normals; a positive slipping in would poison it."""
    tables, _ = build_hardness_tables(teachers, folds)
    for table in tables.values():
        assert (table["label"] == 0).all()


def test_hard_fraction_is_about_one_quarter(teachers, folds):
    """Each fold should mark close to a quarter of its training normals."""
    _, summary = build_hardness_tables(teachers, folds)
    assert ((summary["hard_fraction"] - 0.25).abs() < 0.02).all()


def test_cutoff_uses_only_the_folds_training_groups(teachers, folds):
    """A group outside the training split must not appear in that fold's table."""
    tables, _ = build_hardness_tables(teachers, folds)
    for index, manifest in enumerate(folds):
        training = set(manifest.loc[manifest["split"] == "train", "group_id"])
        assert set(tables[index]["group_id"]) <= training


def test_folds_produce_different_cutoffs(teachers, folds):
    """Different training splits should not all land on one cutoff.

    Identical cutoffs across folds would mean the ranking was computed over a
    shared pool rather than per fold.
    """
    _, summary = build_hardness_tables(teachers, folds)
    assert summary["hardness_cutoff"].nunique() > 1


def test_hard_groups_score_higher_under_both_teachers(teachers, folds):
    """The flag should agree with each model separately, not just their mean."""
    tables, summary = build_hardness_tables(teachers, folds)
    for _, row in summary.iterrows():
        assert row["median_a_hard"] > row["median_a_other"]
        assert row["median_b_hard"] > row["median_b_other"]


def test_flag_covers_every_image_in_a_group(teachers, folds):
    """All images of a hard group must carry the flag, or none of them."""
    tables, _ = build_hardness_tables(teachers, folds)
    manifest = pd.DataFrame({
        "group_id": ["normal:0"] * 3 + ["normal:1"] * 2 + ["pneumonia:0"]})
    table = tables[0]
    mask = flag_images(manifest, table)
    for group in manifest["group_id"].unique():
        rows = mask[manifest["group_id"].to_numpy() == group]
        assert rows.all() or (~rows).all()


def test_empty_training_normals_is_rejected(teachers):
    """A fold with no training normals should fail loudly."""
    with pytest.raises(ValueError, match="không có group NORMAL"):
        hardness_for_fold(teachers, ["pneumonia:0"])


def _weighted_loss(logits, labels, class_weights, multiplier):
    """The Stage B objective: reweight relative emphasis, not overall scale.

    Normalising by the summed multipliers is not enough. PyTorch's weighted
    cross-entropy already divides by the summed class weights rather than by
    the batch size, so dividing by the multipliers alone rescales the loss by
    mean(class_weight) even when every multiplier is one. The denominator has
    to be the summed combined weight.
    """
    per_sample = F.cross_entropy(logits, labels, weight=class_weights,
                                 reduction="none")
    combined = class_weights[labels] * multiplier
    return (per_sample * multiplier).sum() / combined.sum()


def test_all_ones_multiplier_matches_plain_weighted_loss():
    """With no hard negatives the objective must be unchanged."""
    torch.manual_seed(0)
    logits = torch.randn(32, 2)
    labels = torch.randint(0, 2, (32,))
    weights = torch.tensor([1.9, 0.7])
    plain = F.cross_entropy(logits, labels, weight=weights)
    ones = torch.ones(32)
    assert torch.allclose(_weighted_loss(logits, labels, weights, ones),
                          plain, atol=1e-6)


def test_normalising_by_multiplier_sum_keeps_the_loss_scale():
    """Dividing by the batch size instead would inflate the gradient.

    Emphasis on hard negatives is the intended change; a larger effective
    learning rate is not, and would confound the experiment.
    """
    torch.manual_seed(1)
    logits = torch.randn(64, 2)
    labels = torch.randint(0, 2, (64,))
    weights = torch.tensor([1.9, 0.7])
    multiplier = torch.where(torch.arange(64) < 16,
                             torch.full((64,), 2.0), torch.ones(64))

    per_sample = F.cross_entropy(logits, labels, weight=weights,
                                 reduction="none")
    correct = _weighted_loss(logits, labels, weights, multiplier)
    by_multiplier = (per_sample * multiplier).sum() / multiplier.sum()
    plain = F.cross_entropy(logits, labels, weight=weights)

    assert abs(correct - plain) < abs(by_multiplier - plain)


def test_hard_samples_carry_more_gradient():
    """The multiplier must actually shift emphasis onto the flagged samples."""
    torch.manual_seed(2)
    weights = torch.tensor([1.0, 1.0])
    labels = torch.zeros(8, dtype=torch.long)
    multiplier = torch.cat([torch.full((4,), 2.0), torch.ones(4)])

    logits = torch.randn(8, 2, requires_grad=True)
    _weighted_loss(logits, labels, weights, multiplier).backward()
    grad = logits.grad.abs().sum(dim=1)
    assert grad[:4].mean() > grad[4:].mean()
