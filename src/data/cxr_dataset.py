from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PathLike = Union[str, Path]


class CXRDataset(Dataset):
    """Dataset backed by a dataframe with at least ``path`` and ``label``.

    Paths may be absolute or relative to ``root_dir``. Images are opened as
    grayscale PIL images. The transform decides whether the final tensor has
    one channel or three repeated grayscale channels.
    """

    REQUIRED_COLUMNS = {"path", "label"}

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        root_dir: Optional[PathLike] = None,
        verify_paths: bool = False,
        return_metadata: bool = False,
        metadata_columns: Optional[Iterable[str]] = None,
    ) -> None:
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir else None
        self.return_metadata = return_metadata
        self.metadata_columns = list(metadata_columns or ["path"])

        labels = pd.to_numeric(self.df["label"], errors="raise").astype(int)
        if (labels < 0).any():
            raise ValueError("Labels must be non-negative integers.")
        self.df["label"] = labels

        if verify_paths:
            missing_paths = [str(path) for path in self.resolved_paths if not path.is_file()]
            if missing_paths:
                preview = missing_paths[:5]
                raise FileNotFoundError(
                    f"{len(missing_paths)} image paths do not exist. First examples: {preview}"
                )

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> list[int]:
        return self.df["label"].astype(int).tolist()

    @property
    def resolved_paths(self) -> list[Path]:
        return [self._resolve_path(value) for value in self.df["path"].tolist()]

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            if self.root_dir is None:
                raise ValueError(
                    f"Relative path '{path}' found but root_dir was not provided."
                )
            path = self.root_dir / path
        return path.resolve()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self._resolve_path(row["path"])

        try:
            with Image.open(image_path) as source:
                image = source.convert("L")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        if not self.return_metadata:
            return image, label

        metadata = {
            column: row[column]
            for column in self.metadata_columns
            if column in self.df.columns
        }
        metadata["resolved_path"] = str(image_path)
        metadata["index"] = int(idx)
        return image, label, metadata
