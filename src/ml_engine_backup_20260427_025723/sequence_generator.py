"""
src/ml_engine/sequence_generator.py — The Memory Guard.

Role: Provide a PyTorch-compatible Dataset that transforms a flat 2D
feature matrix into sliding-window 3D sequences on demand, preventing
Out-of-Memory errors that would occur if all windows were materialised
at once.

Design rationale
----------------
Deep Learning models (LSTM, GRU, TCN, CNN-LSTM, TransformerEncoder)
require input of shape (batch_size, window_size, n_features). For a
2-year M1 history of 700,000 bars with window_size=30 and 50 features,
fully materialising all windows in RAM requires approximately 4.2 GB.
This class avoids that by storing only the flat (N, F) matrix and
slicing windows lazily inside __getitem__, which PyTorch's DataLoader
calls one batch at a time during training.

Window-label alignment
-----------------------
For a window ending at index t (i.e. bars t-window_size+1 through t),
the label is the outcome at bar t. This means the model sees the last
window_size bars and predicts what happens at the end of that window.
The Labeler must be run on the same bar DataFrame before constructing
this Dataset — the label at index t must reflect the price movement
starting from bar t.

Index alignment
---------------
FeatureMatrix and the label Series produced by Labeler.compute_labels()
may have different lengths. This Dataset takes a FeatureMatrix and a
pd.Series and aligns them on their shared index before extracting the
numpy arrays. Any row present in features but absent in labels is
silently excluded during the intersection step.

Time-series shuffle warning
-----------------------------
shuffle=True on the training DataLoader breaks temporal order and can
introduce subtle data leakage in walk-forward validation regimes. The
get_dataloader() factory enforces shuffle=False by default and logs a
prominent warning if the caller explicitly requests shuffle=True.
DataShaper enforces the time-ordered train/val/test split upstream —
this class is the last line of defence.

Public API
----------
    TimeSeriesDataset(feature_matrix, labels, window_size)
    TimeSeriesDataset.__len__()                -> int
    TimeSeriesDataset.__getitem__(idx)         -> tuple[Tensor, Tensor]
    get_dataloader(feature_matrix, labels, window_size, batch_size,
                    shuffle, num_workers)       -> DataLoader
"""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_engine.features import FeatureMatrix

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ── Module Constants ─────────────────────────────────────────────────────────

# Minimum window size enforced at construction time. A window of fewer
# than 2 bars cannot encode any temporal relationship.
_MIN_WINDOW_SIZE: int = 2

# Minimum viable dataset length after alignment and windowing. A dataset
# with fewer samples than this produces unreliable gradient estimates.
_MIN_VIABLE_SAMPLES: int = 10


# ── Custom Exception ─────────────────────────────────────────────────────────


class SequenceGeneratorError(Exception):
    """
    Raised when TimeSeriesDataset cannot produce a valid sequence set.

    Distinct from ValueError (caller contract violation such as a
    malformed input) — SequenceGeneratorError signals a runtime failure
    inside the windowing pipeline that the trainer must handle explicitly.

    Attributes:
        stage: The pipeline stage that failed (e.g. "init", "getitem").
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SequenceGeneratorError(stage={self.stage!r}, " f"message={str(self)!r})"
        )


# ── TimeSeriesDataset ────────────────────────────────────────────────────────


class TimeSeriesDataset(Dataset):  # type: ignore[type-arg]
    """
    PyTorch Dataset for on-the-fly sliding-window feature sequences.

    Accepts a FeatureMatrix (shape N x F) and a label Series, aligns
    them on their shared index, and yields (window_tensor, label_tensor)
    pairs lazily during training. Only one batch occupies RAM at any
    time — the full window set is never materialised.

    Input shapes
    ------------
        feature_matrix.matrix : (N, F)  — N bars, F features
        labels                : (M,)    — M <= N after Labeler tail-drop

    Output shapes per __getitem__ call
    -----------------------------------
        x : (window_size, F)  — float32 tensor
        y : ()                — scalar float32 tensor (0.0 or 1.0)

    The number of valid windows is len(aligned_rows) - window_size.
    Window i covers rows i through i + window_size - 1 inclusive.
    The label for window i is the label at row i + window_size - 1
    (the outcome at the last bar in the window).

    Attributes:
        window_size:   Number of consecutive bars per sequence.
        n_features:    Number of features per bar (F dimension).
        n_samples:     Total number of valid windows in the dataset.
        feature_version: Feature engineering version from FeatureMatrix,
                        stored for trainer metadata validation.

    Example:
        >>> dataset = TimeSeriesDataset(feature_matrix, labels, window_size=30)
        >>> len(dataset)
        699970
        >>> x, y = dataset[0]
        >>> x.shape
        torch.Size([30, 50])
        >>> y.shape
        torch.Size([])
    """

    def __init__(
        self,
        feature_matrix: FeatureMatrix,
        labels: pd.Series,
        window_size: int = 30,
    ) -> None:
        """
        Initialise the dataset and align features to labels.

        Performs index alignment between the FeatureMatrix timestamps
        and the label Series index, then stores the aligned numpy arrays
        for lazy slicing. All validation and alignment happens here so
        that __getitem__ is a pure indexing operation with no overhead.

        Args:
            feature_matrix: FeatureMatrix produced by
                FeatureEngineer.build_matrix(). Carries the version
                string and symbol for metadata validation.
            labels: pd.Series produced by Labeler.compute_labels().
                Index must overlap with feature_matrix.timestamps.
                Series name should be "label" (set automatically by
                Labeler).
            window_size: Number of consecutive M1 bars per sequence.
                Must be >= _MIN_WINDOW_SIZE (2). Larger windows give
                the model more temporal context but reduce the number
                of valid samples. Recommended range: 20-60 for M1 data.

        Raises:
            ValueError: If window_size < _MIN_WINDOW_SIZE, or if
                feature_matrix is not a FeatureMatrix instance.
            SequenceGeneratorError: If the aligned dataset has fewer
                than window_size + _MIN_VIABLE_SAMPLES rows, or if
                the feature/label index intersection is empty.
        """
        import pandas as pd  # local import — pandas is heavy, keep at call site

        if not isinstance(feature_matrix, FeatureMatrix):
            raise ValueError(
                f"[!] feature_matrix must be a FeatureMatrix instance, "
                f"got {type(feature_matrix).__name__}. "
                f"Use FeatureEngineer.build_matrix() to produce one."
            )

        if window_size < _MIN_WINDOW_SIZE:
            raise ValueError(
                f"[%] window_size={window_size} is below the minimum of "
                f"{_MIN_WINDOW_SIZE}. A window of fewer than 2 bars cannot "
                f"encode any temporal relationship."
            )

        # ── Index Alignment ──────────────────────────────────────────────────
        # FeatureMatrix.timestamps is a list of pd.Timestamp objects.
        # Build a DataFrame indexed by those timestamps so we can align
        # with the label Series on their shared index.
        feat_index = pd.DatetimeIndex(feature_matrix.timestamps)
        label_index = pd.DatetimeIndex(labels.index)

        common_index = feat_index.intersection(label_index)

        if common_index.empty:
            raise SequenceGeneratorError(
                "[!] Feature and label indices have no common timestamps. "
                "Ensure Labeler.compute_labels() was called on the same bar "
                "DataFrame that produced the FeatureMatrix.",
                stage="init",
            )

        # Locate positions in the feature matrix array for the common index.
        # feat_index.get_indexer returns integer positions (-1 = not found).
        feat_positions = feat_index.get_indexer(common_index)
        if (feat_positions == -1).any():
            raise SequenceGeneratorError(
                "[%] Index alignment produced missing positions in the feature "
                "matrix. This should not occur after intersection — verify "
                "that feature_matrix.timestamps has no duplicates.",
                stage="init",
            )

        # Extract aligned arrays. Both are float32 — labels are stored as
        # float32 so BCELoss / BCEWithLogitsLoss receives the correct dtype.
        self._features: np.ndarray = feature_matrix.matrix[feat_positions].astype(
            np.float32
        )
        self._labels: np.ndarray = np.asarray(
            labels.loc[common_index].values, dtype=np.float32
        )

        aligned_rows: int = len(self._features)
        n_samples: int = aligned_rows - window_size

        if n_samples < _MIN_VIABLE_SAMPLES:
            raise SequenceGeneratorError(
                f"[%] Dataset too small after alignment and windowing. "
                f"Aligned rows={aligned_rows}, window_size={window_size}, "
                f"valid samples={n_samples} (minimum={_MIN_VIABLE_SAMPLES}). "
                f"Provide a longer historical bar range.",
                stage="init",
            )

        self.window_size: int = window_size
        self.n_features: int = self._features.shape[1]
        self.n_samples: int = n_samples
        self.feature_version: str = feature_matrix.version
        self.symbol: str = feature_matrix.symbol

        logger.info(
            "[^] TimeSeriesDataset ready: symbol=%s aligned_rows=%d "
            "window_size=%d n_samples=%d n_features=%d version=%s",
            self.symbol,
            aligned_rows,
            self.window_size,
            self.n_samples,
            self.n_features,
            self.feature_version,
        )

    # ── PyTorch Dataset Interface ────────────────────────────────────────────

    def __len__(self) -> int:
        """
        Return the total number of valid windows in the dataset.

        This is len(aligned_rows) - window_size. PyTorch's DataLoader
        uses this value to determine batch boundaries and epoch length.

        Returns:
            int: Number of valid (window, label) pairs.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single (window, label) pair by index.

        Slices the pre-aligned numpy array to extract the window of
        shape (window_size, n_features) starting at idx, and the
        scalar label at position idx + window_size - 1. Conversion
        to torch.Tensor happens here — no copying occurs because
        torch.from_numpy shares the underlying memory buffer.

        Args:
            idx: Integer index in [0, n_samples). PyTorch's DataLoader
                passes valid indices based on __len__ — out-of-bounds
                access is the DataLoader's responsibility.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                x: float32 tensor of shape (window_size, n_features).
                    Represents window_size consecutive feature bars.
                y: scalar float32 tensor (0.0 or 1.0).
                    Represents the binary outcome at the end of the
                    window — 1.0 for CALL win, 0.0 for PUT win.

        Raises:
            SequenceGeneratorError: If the extracted window contains
                non-finite values (NaN or Inf) that would corrupt
                gradient computation.
        """
        # Window: rows idx through idx + window_size - 1 inclusive.
        window: np.ndarray = self._features[idx : idx + self.window_size]

        # Guard: non-finite values in a training batch corrupt gradients
        # silently. Catch them here where the index is known.
        if not np.isfinite(window).all():
            bad_positions = np.argwhere(~np.isfinite(window))
            raise SequenceGeneratorError(
                f"[%] Non-finite values detected in window at dataset "
                f"idx={idx}. Positions (bar, feature): {bad_positions.tolist()}. "
                f"Re-run FeatureEngineer.build_matrix() and verify the source "
                f"bar data has no gaps beyond _MAX_FFILL_BARS.",
                stage="getitem",
            )

        # Label: outcome at the final bar of the window.
        label: np.floating[Any] = self._labels[idx + self.window_size - 1]

        # torch.from_numpy shares memory — no copy, zero overhead.
        x_tensor: torch.Tensor = torch.from_numpy(window)
        y_tensor: torch.Tensor = torch.tensor(label, dtype=torch.float32)

        return x_tensor, y_tensor

    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataset("
            f"symbol={self.symbol!r}, "
            f"n_samples={self.n_samples}, "
            f"window_size={self.window_size}, "
            f"n_features={self.n_features}, "
            f"version={self.feature_version!r})"
        )


# ── DataLoader Factory ───────────────────────────────────────────────────────


def get_dataloader(
    feature_matrix: FeatureMatrix,
    labels: pd.Series,
    window_size: int = 30,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:  # type: ignore[type-arg]
    """
    Construct a PyTorch DataLoader wrapping a TimeSeriesDataset.

    This is the primary entry point for Deep Learning trainers. It
    builds the TimeSeriesDataset, applies index alignment, and returns
    a DataLoader configured for time-series training. Callers should
    use this function rather than constructing DataLoader directly.

    Shuffle behaviour
    -----------------
    shuffle=False is the default and the correct setting for all
    validation and test splits. For training splits, shuffle=True is
    acceptable because DataShaper has already enforced a time-ordered
    train/val/test boundary — shuffling within the training window
    does not leak future data into the past. A WARNING is logged
    when shuffle=True to make this explicit in every training run.

    pin_memory behaviour
    --------------------
    pin_memory=True is set automatically when a CUDA GPU is available,
    enabling faster host-to-device tensor transfers during training.
    It is disabled on CPU-only environments to avoid unnecessary
    memory pinning overhead.

    Args:
        feature_matrix: FeatureMatrix from FeatureEngineer.build_matrix().
        labels:         pd.Series from Labeler.compute_labels().
        window_size:    Bars per sequence window. Default 30 (30 minutes
                        of M1 context). Must match the window size used
                        during training when loading for inference.
        batch_size:     Number of windows per gradient update step.
                        Default 64. Larger batches require more RAM but
                        produce more stable gradients.
        shuffle:        Whether to shuffle the dataset each epoch.
                        Use False for val/test, True only for training
                        within a time-ordered split.
        num_workers:    Number of subprocess workers for data loading.
                        Default 0 (main process only). Set to 2-4 on
                        multi-core training machines. Keep at 0 on
                        Azure Container Instances to avoid fork issues.

    Returns:
        DataLoader: Configured PyTorch DataLoader ready for iteration
            in a training loop. Each iteration yields
            (x, y) where x.shape == (batch_size, window_size, n_features)
            and y.shape == (batch_size,).

    Raises:
        ValueError:              If window_size < _MIN_WINDOW_SIZE.
        SequenceGeneratorError:  If the dataset cannot be constructed
                                    (see TimeSeriesDataset.__init__).
    """
    if shuffle:
        warning_block = (
            f"\n{'%' * 60}\n"
            f"WARNING: shuffle=True is enabled for this DataLoader. "
            f"Ensure this DataLoader is used for the TRAINING split only. "
            f"Shuffling validation or test splits breaks time-series "
            f"evaluation integrity and can lead to misleading performance "
            f"metrics."
            f"\n{'%' * 60}\n"
        )
        logger.warning(warning_block)

    dataset = TimeSeriesDataset(
        feature_matrix=feature_matrix,
        labels=labels,
        window_size=window_size,
    )

    # Windows uses multiprocessing spawn for DataLoader workers, which pickles
    # the entire Dataset. If any field is unpickleable (open file handles, etc.)
    # workers crash silently. Force num_workers=0 on Windows to prevent this.
    if platform.system() == "Windows" and num_workers > 0:
        logger.warning(
            "[%%] Windows detected: forcing num_workers=0 for DataLoader. "
            "Multiprocessing spawn is unreliable for this Dataset on Windows. "
            "Pass num_workers=0 explicitly to suppress this warning."
        )
        num_workers = 0

    # pin_memory speeds up CPU -> GPU transfer but has no effect and
    # wastes pinned memory pages on a CPU-only machine.
    use_pin_memory: bool = torch.cuda.is_available()

    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    info_block = (
        f"\n{'^' * 60}\n"
        f"DataLoader configuration: "
        f"symbol={dataset.symbol} "
        f"n_samples={dataset.n_samples} "
        f"batch_size={batch_size} "
        f"window_size={window_size} "
        f"shuffle={shuffle} "
        f"pin_memory={use_pin_memory} "
        f"num_workers={num_workers}"
        f"\n{'^' * 60}"
    )
    logger.info(info_block)

    return loader
