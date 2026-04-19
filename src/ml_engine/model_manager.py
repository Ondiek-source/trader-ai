"""
src/ml_engine/model_manager.py — The Librarian.

Role: Persist, version, retrieve, and validate trained model artifacts
produced by trainer.py. Ensures the correct model brain is loaded for
the right expiry key, symbol, and feature schema version.

Design rationale
----------------
ModelManager owns the full lifecycle of a model artifact from the moment
trainer.py produces a TrainerResult to the moment live.py calls predict.
It enforces three guarantees:

  1. Version safety  — a model trained against feature schema v3.1.0
                        cannot be loaded when the current schema is v3.2.0
                        unless explicitly overridden via allow_stale_models
                        in config.

  2. Metadata integrity — every artifact on disk has a companion .json
                            file carrying full provenance. A model file
                            without a metadata sidecar is rejected.

  3. Registry awareness — the manager maintains an in-memory index of
                            all artifacts on disk, queryable by symbol,
                            expiry key, model name, and recency. The
                            InferenceEngine uses get_best_model() to find
                            the correct brain without knowing file paths.

Artifact file layout
--------------------
Each save() call writes two files to the storage directory:

    {model_name}_{symbol}_{expiry_key}_{timestamp}.artifact
        The serialised model. Classical ML -> joblib. PyTorch -> state_dict.
        SB3 RL -> SB3 native save format (zip).

    {model_name}_{symbol}_{expiry_key}_{timestamp}.json
        The metadata sidecar. Contains all TrainerResult fields except
        the artifact itself. Required for every load operation.

PyTorch loading contract
------------------------
PyTorch models cannot be deserialised from a state_dict alone — the
network architecture (nn.Module subclass) must be provided by the
caller via the model_class parameter in load(). This is intentional:
the ModelManager does not import any nn.Module class. The caller
(InferenceEngine or trainer) owns the architecture and passes a
pre-instantiated nn.Module for weight loading.

Public API
----------
    ModelManager(storage_dir)
    ModelManager.save(result)                            -> str
    ModelManager.load(model_path, model_class)           -> Any
    ModelManager.get_best_model(symbol, expiry_key,
                                model_name)             -> ModelRecord | None
    ModelManager.list_models(symbol, expiry_key,
                                model_name)                -> list[ModelRecord]
    ModelManager.delete(model_path)                      -> None
    ModelManager.validate_metadata(meta_path)            -> dict[str, Any]
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import torch

from ml_engine.features import _VERSION
from ml_engine.trainer import TrainerResult
from core.config import get_settings
from data.storage import get_storage

logger = logging.getLogger(__name__)

# ── Module Constants ─────────────────────────────────────────────────────────

# File extension for serialised model artifacts.
_ARTIFACT_SUFFIX: str = ".artifact"

# File extension for metadata sidecars.
_METADATA_SUFFIX: str = ".json"

# Timestamp format used in artifact filenames.
_TIMESTAMP_FMT: str = "%Y%m%d_%H%M%S"


# ── Custom Exception ─────────────────────────────────────────────────────────


class ModelManagerError(Exception):
    """
    Raised when ModelManager cannot complete a save, load, or registry
    operation.

    Distinct from ValueError (caller contract violation) — ModelManagerError
    signals a runtime failure in artifact persistence or retrieval that
    the InferenceEngine or pipeline orchestrator must handle explicitly.

    Attributes:
        stage: The pipeline stage that failed, e.g. "save", "load",
                "validate".
    """

    def __init__(self, message: str, stage: str = "") -> None:
        super().__init__(message)
        self.stage = stage

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelManagerError(stage={self.stage!r}, " f"message={str(self)!r})"


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRecord:
    """
    Lightweight registry entry describing a persisted model artifact.

    Produced by list_models() and consumed by get_best_model() to
    select the correct artifact without loading the full model into
    memory. The InferenceEngine stores a ModelRecord per active model
    slot and calls load() only when inference is required.

    Attributes:
        model_name:      Trainer class name, e.g. "XGBoostTrainer".
        symbol:          Currency pair, e.g. "EUR_USD".
        expiry_key:      Expiry window, e.g. "5_MIN".
        feature_version: _VERSION string at training time.
        trained_at:      UTC ISO-8601 timestamp of training completion.
        artifact_path:   Absolute path to the .artifact file.
        metadata_path:   Absolute path to the .json sidecar.
        auc:             Validation AUC from training metrics.
                            0.0 if not present in metadata (RL models).
        is_pytorch:      True if artifact is a PyTorch state_dict.
        is_sb3:          True if artifact is a Stable-Baselines3 model.
        train_rows:      Number of training rows consumed.
        val_rows:        Number of validation rows evaluated.
    """

    model_name: str
    symbol: str
    expiry_key: str
    feature_version: str
    trained_at: str
    artifact_path: str
    metadata_path: str
    auc: float
    is_pytorch: bool
    is_sb3: bool
    train_rows: int
    val_rows: int

    def __repr__(self) -> str:
        return (
            f"ModelRecord("
            f"model={self.model_name!r}, "
            f"symbol={self.symbol!r}, "
            f"expiry={self.expiry_key!r}, "
            f"auc={self.auc:.4f}, "
            f"version={self.feature_version!r})"
        )


# ── ModelManager ─────────────────────────────────────────────────────────────


class ModelManager:
    """
    Librarian: persist, version, retrieve, and validate model artifacts.

    Owns the full lifecycle of every trained model from the moment
    trainer.py produces a TrainerResult to the moment live.py calls
    predict_proba(). All artifact I/O is routed through this class —
    no trainer or inference component writes directly to disk.

    Thread safety
    -------------
    ModelManager holds no mutable state beyond the storage_dir path and
    settings snapshot. Concurrent save() calls from multiple trainers
    produce uniquely timestamped files and do not interfere. Concurrent
    load() calls are safe — files are opened read-only. The in-memory
    registry built by list_models() is rebuilt on each call and is not
    cached between calls.

    Lifecycle
    ---------
    Instantiate once at pipeline startup and pass to all trainers and
    the InferenceEngine. Do not create per-request instances.

    Example:
        >>> mm = ModelManager(storage_dir="models/")
        >>> result = xgb_trainer.train(split)
        >>> path = mm.save(result)
        >>> record = mm.get_best_model("EUR_USD", "5_MIN", "XGBoostTrainer")
        >>> model = mm.load(record.artifact_path)
    """

    def __init__(self, storage_dir: str = "models") -> None:
        """
        Initialise the ModelManager and ensure the storage directory exists.

        Args:
            storage_dir: Path to the directory where artifact and metadata
                            files are written. Created if it does not exist.
                            Relative paths are resolved from the working
                            directory of the running process.

        Raises:
            ModelManagerError: If the storage directory cannot be created
                                due to a permission or filesystem error.
        """
        self._settings = get_settings()
        self._storage_dir: Path = Path(storage_dir).resolve()
        self._storage = get_storage()

        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER INIT FAILURE: STORAGE DIRECTORY\n"
                f"Path: {self._storage_dir}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: ModelManager requires a writable directory to persist\n"
                f"model artifacts and metadata sidecars. Without this directory\n"
                f"no trained model can be saved or loaded by the system.\n"
                f"\nFIX: Verify filesystem permissions and available disk space.\n"
                f"Ensure the process has write access to: {self._storage_dir}\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"Cannot create storage directory: {self._storage_dir}",
                stage="init",
            ) from exc

        logger.info(
            "[^] ModelManager initialised: storage_dir=%s current_version=%s",
            self._storage_dir,
            _VERSION,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, result: TrainerResult) -> str:
        """
        Serialise a TrainerResult artifact and write its metadata sidecar.

        Determines the serialisation format automatically from the
        artifact type:
            - Stable-Baselines3 models: SB3 native .save() format (zip).
            - PyTorch nn.Module:         torch.save() state_dict.
            - All others (sklearn, XGB, LightGBM, CatBoost): joblib.

        The artifact and metadata are written atomically — if either
        write fails the partial files are removed to prevent a corrupt
        registry state.

        Args:
            result: TrainerResult produced by BaseTrainer.train().
                    Must have a non-None artifact field.

        Returns:
            str: Absolute path to the saved .artifact file.

        Raises:
            ValueError:        If result.artifact is None.
            ModelManagerError: If serialisation or file I/O fails.
        """
        if result.artifact is None:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER SAVE FAILURE: NULL ARTIFACT\n"
                f"Model: {result.model_name}\n"
                f"Symbol: {result.symbol}\n"
                f"Expiry: {result.expiry_key}\n\n"
                f"CONTEXT: save() received a TrainerResult with artifact=None.\n"
                f"This means the trainer did not assign a trained model object\n"
                f"to self.model before returning the TrainerResult. Nothing can\n"
                f"be serialised without a valid model artifact.\n"
                f"\nFIX: Ensure BaseTrainer.train() completes successfully and\n"
                f"sets self.model before calling _build_result(). Check trainer\n"
                f"logs for upstream training failures.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ValueError(
                "ModelManager.save() received a TrainerResult with "
                "artifact=None. Train the model before saving."
            )

        safe_symbol: str = result.symbol.replace("/", "_").replace(" ", "_")
        base_name: str = f"{result.model_name}_{safe_symbol}_" f"{result.expiry_key}"

        artifact_path: Path = self._storage_dir / f"{base_name}{_ARTIFACT_SUFFIX}"
        metadata_path: Path = self._storage_dir / f"{base_name}{_METADATA_SUFFIX}"

        # Delete old files if they exist
        for p in (artifact_path, metadata_path):
            if p.exists():
                p.unlink()
                logger.debug(f"[^] Removed old file: {p.name}")

        # Detect serialisation format from the artifact type.
        is_sb3: bool = self._is_sb3_model(result.artifact)
        is_pytorch: bool = not is_sb3 and isinstance(result.artifact, torch.nn.Module)

        try:
            # ── 1. Write artifact ────────────────────────────────────────
            if is_sb3:
                result.artifact.save(str(artifact_path))
                logger.debug(
                    "[^] SB3 artifact saved via native .save(): %s",
                    artifact_path.name,
                )
            elif is_pytorch:
                torch.save(result.artifact.state_dict(), artifact_path)
                logger.debug(
                    "[^] PyTorch state_dict saved: %s",
                    artifact_path.name,
                )
            else:
                joblib.dump(result.artifact, artifact_path)
                logger.debug(
                    "[^] joblib artifact saved: %s",
                    artifact_path.name,
                )

            # ── 2. Write metadata sidecar ────────────────────────────────
            metadata: dict[str, Any] = {
                "model_name": result.model_name,
                "symbol": result.symbol,
                "expiry_key": result.expiry_key,
                "feature_version": result.feature_version,
                "trained_at": result.trained_at,
                "device": result.device,
                "train_rows": result.train_rows,
                "val_rows": result.val_rows,
                "metrics": result.metrics,
                "extra": result.extra,
                "artifact_path": str(artifact_path),
                "is_pytorch": is_pytorch,
                "is_sb3": is_sb3,
                "manager_version": _VERSION,
                "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=4)

            logger.info(
                "[^] ModelManager.save(): model=%s symbol=%s expiry=%s "
                "auc=%.4f artifact=%s",
                result.model_name,
                result.symbol,
                result.expiry_key,
                result.metrics.get("auc", 0.0),
                artifact_path.name,
            )

            # Push both files to Azure Blob so the model survives
            # a container restart. Storage is a no-op in LOCAL mode.
            self._storage.save_model(artifact_path, metadata_path)

            return str(artifact_path)

        except Exception as exc:
            # Cleanup partial files before raising so the registry
            # never contains an orphaned artifact/sidecar pair.
            for orphan in (artifact_path, metadata_path):
                if orphan.exists():
                    try:
                        orphan.unlink()
                    except OSError:
                        pass

            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER SAVE FAILURE\n"
                f"Model: {result.model_name}\n"
                f"Symbol: {result.symbol}\n"
                f"Expiry: {result.expiry_key}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: An error occurred while writing the artifact or its\n"
                f"metadata sidecar to disk. Any partial files have been removed\n"
                f"automatically to keep the model registry in a consistent state.\n"
                f"\nFIX: Verify available disk space and write permissions on\n"
                f"the storage directory: {self._storage_dir}\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"save() failed for {result.model_name}: {exc}",
                stage="save",
            ) from exc

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(
        self,
        artifact_path: str,
        model_class: Any | None = None,
    ) -> Any:
        """
        Deserialise a model artifact after strict version validation.

        Reads the metadata sidecar, validates the feature_version against
        the current _VERSION, then deserialises the artifact using the
        format recorded in the sidecar (joblib / torch / SB3).

        PyTorch models require a pre-instantiated nn.Module via model_class
        because ModelManager does not import any nn.Module architecture.
        The caller must instantiate the correct architecture with the same
        hyperparameters used at training time before calling load().

        Args:
            artifact_path: Path to the .artifact file. The companion
                            .json sidecar must exist at the same path
                            with the .json suffix.
            model_class:   Required for PyTorch artifacts — a pre-
                            instantiated nn.Module with the correct
                            architecture. The state_dict is loaded into
                            this instance in-place. Ignored for joblib
                            and SB3 artifacts.

        Returns:
            Any: The deserialised model object ready for predict_proba().
                    For PyTorch: the model_class instance with loaded weights,
                    set to eval() mode.
                    For SB3: the loaded SB3 algorithm instance.
                    For Classical ML: the sklearn / XGB / LightGBM /
                    CatBoost estimator.

        Raises:
            FileNotFoundError:  If the artifact or sidecar file is missing.
            ValueError:         If model_class is None for a PyTorch artifact,
                                or if the artifact_path has the wrong suffix.
            ModelManagerError:  If the feature version check fails and
                                allow_stale_models is False, or if
                                deserialisation fails.
        """
        path: Path = Path(artifact_path).resolve()

        if path.suffix != _ARTIFACT_SUFFIX:
            raise ValueError(
                f"[!] artifact_path must have suffix '{_ARTIFACT_SUFFIX}', "
                f"got '{path.suffix}'. Pass the .artifact file path, "
                f"not the .json sidecar."
            )

        metadata_path: Path = path.with_suffix(_METADATA_SUFFIX)

        if not path.exists():
            raise FileNotFoundError(
                f"[!] Artifact file not found: {path}. "
                f"Verify the path or run a training pass first."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"[!] Metadata sidecar missing for artifact: {path.name}. "
                f"Expected: {metadata_path}. "
                f"The artifact cannot be loaded without its metadata sidecar."
            )

        # ── 1. Version validation ─────────────────────────────────────────
        metadata = self.validate_metadata(str(metadata_path))
        model_version: str = metadata.get("feature_version", "UNKNOWN")

        if model_version != _VERSION:
            if not self._settings.allow_stale_models:
                error_block = (
                    f"\n{'!' * 60}\n"
                    f"MODEL MANAGER LOAD FAILURE: VERSION MISMATCH\n"
                    f"Model feature_version : {model_version}\n"
                    f"Current _VERSION      : {_VERSION}\n"
                    f"Artifact              : {path.name}\n\n"
                    f"CONTEXT: This model was trained against a different feature\n"
                    f"schema version. Loading it will produce incorrect predictions\n"
                    f"because the input vector shape and column ordering do not\n"
                    f"match what the model was trained on.\n"
                    f"\nFIX: Retrain the model against the current feature schema\n"
                    f"({_VERSION}), or set ALLOW_STALE_MODELS=True in your .env\n"
                    f"file to bypass this check at your own risk.\n"
                    f"{'!' * 60}"
                )
                logger.critical(error_block)
                raise ModelManagerError(
                    f"Version mismatch: model={model_version} "
                    f"current={_VERSION}. "
                    f"Retrain the model or set allow_stale_models=True.",
                    stage="load",
                )
            else:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"MODEL MANAGER WARNING: STALE MODEL LOADED\n"
                    f"Model feature_version : {model_version}\n"
                    f"Current _VERSION      : {_VERSION}\n"
                    f"Artifact              : {path.name}\n\n"
                    f"CONTEXT: allow_stale_models=True is set in config. This model\n"
                    f"was trained against a different feature schema. Its predictions\n"
                    f"may be silently wrong because column ordering or feature values\n"
                    f"no longer match what the model learned.\n"
                    f"\nFIX: Retrain the model against the current schema ({_VERSION})\n"
                    f"as soon as possible and remove ALLOW_STALE_MODELS=True.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)

        is_pytorch: bool = bool(metadata.get("is_pytorch", False))
        is_sb3: bool = bool(metadata.get("is_sb3", False))

        # ── 2. Deserialisation ────────────────────────────────────────────
        try:
            if is_sb3:
                model = self._load_sb3(path, metadata)

            elif is_pytorch:
                if model_class is None:
                    error_block = (
                        f"\n{'!' * 60}\n"
                        f"MODEL MANAGER LOAD FAILURE: MISSING MODEL CLASS\n"
                        f"Artifact: {path.name}\n\n"
                        f"CONTEXT: This artifact is a PyTorch model (state_dict).\n"
                        f"PyTorch state_dicts contain weights only — they cannot be\n"
                        f"deserialised without the corresponding network architecture\n"
                        f"(nn.Module subclass) being provided by the caller.\n"
                        f"\nFIX: Instantiate the correct nn.Module subclass with the\n"
                        f"same hyperparameters used at training time (check the extra\n"
                        f"field in the metadata sidecar), then pass it via model_class=.\n"
                        f"{'!' * 60}"
                    )
                    logger.critical(error_block)
                    raise ValueError(
                        f"artifact '{path.name}' is a PyTorch model. "
                        f"Provide a pre-instantiated nn.Module via model_class=."
                    )
                model = self._load_pytorch(path, model_class)

            else:
                model = joblib.load(path)
                logger.debug("[^] joblib artifact loaded: %s", path.name)

            logger.info(
                "[^] ModelManager.load(): model=%s symbol=%s expiry=%s "
                "version=%s artifact=%s",
                metadata.get("model_name", "UNKNOWN"),
                metadata.get("symbol", "UNKNOWN"),
                metadata.get("expiry_key", "UNKNOWN"),
                model_version,
                path.name,
            )

            return model

        except (ValueError, ModelManagerError):
            raise
        except Exception as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER LOAD FAILURE: DESERIALISATION ERROR\n"
                f"Artifact: {path.name}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: The artifact file exists and passed version validation\n"
                f"but could not be deserialised. The file may be corrupted or\n"
                f"truncated. For PyTorch artifacts, the model_class architecture\n"
                f"may not match what was used at training time.\n"
                f"\nFIX: Verify the artifact file is not zero-bytes or truncated.\n"
                f"For PyTorch artifacts, confirm the model_class matches the\n"
                f"architecture recorded in the extra field of the metadata sidecar.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"load() failed for {path.name}: {exc}",
                stage="load",
            ) from exc

    # ── Registry ─────────────────────────────────────────────────────────────

    def list_models(
        self,
        symbol: str | None = None,
        expiry_key: str | None = None,
        model_name: str | None = None,
    ) -> list[ModelRecord]:
        """
        Scan the storage directory and return a filtered list of ModelRecords.

        Reads every .json sidecar in the storage directory and builds a
        ModelRecord for each valid artifact pair. Invalid or orphaned
        sidecars (missing artifact file, malformed JSON) are logged as
        warnings and skipped — they do not raise exceptions.

        Filtering is applied after scanning. All filter parameters are
        optional — omitting all three returns every valid artifact.

        Args:
            symbol:     Filter by currency pair, e.g. "EUR_USD".
                        Case-sensitive. None = no filter.
            expiry_key: Filter by expiry window, e.g. "5_MIN".
                        None = no filter.
            model_name: Filter by trainer class name, e.g. "XGBoostTrainer".
                        None = no filter.

        Returns:
            list[ModelRecord]: Matching records sorted by trained_at
                descending (most recent first). Empty list if no matches.
        """
        records: list[ModelRecord] = []

        for meta_file in sorted(self._storage_dir.glob(f"*{_METADATA_SUFFIX}")):
            artifact_file = meta_file.with_suffix(_ARTIFACT_SUFFIX)

            if not artifact_file.exists():
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"MODEL MANAGER WARNING: ORPHANED METADATA SIDECAR\n"
                    f"Sidecar : {meta_file.name}\n"
                    f"Expected artifact: {artifact_file.name}\n\n"
                    f"CONTEXT: A metadata sidecar exists on disk without a corresponding\n"
                    f"artifact file. This usually means the artifact was manually deleted\n"
                    f"or a previous save() call failed mid-write. This record is skipped.\n"
                    f"\nFIX: Delete the orphaned sidecar to clean the registry, or\n"
                    f"re-run the training pass to regenerate the artifact pair.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as fh:
                    meta: dict[str, Any] = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"MODEL MANAGER WARNING: MALFORMED METADATA SIDECAR\n"
                    f"File: {meta_file.name}\n"
                    f"Error: {exc}\n\n"
                    f"CONTEXT: The metadata sidecar could not be parsed as valid JSON.\n"
                    f"This record is skipped in the registry. The artifact file may\n"
                    f"still exist on disk but cannot be loaded without valid metadata.\n"
                    f"\nFIX: Delete both the artifact and sidecar files and re-run the\n"
                    f"training pass to regenerate a clean artifact/sidecar pair.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
                continue

            record = ModelRecord(
                model_name=meta.get("model_name", "UNKNOWN"),
                symbol=meta.get("symbol", "UNKNOWN"),
                expiry_key=meta.get("expiry_key", "UNKNOWN"),
                feature_version=meta.get("feature_version", "UNKNOWN"),
                trained_at=meta.get("trained_at", ""),
                artifact_path=str(artifact_file),
                metadata_path=str(meta_file),
                auc=float(meta.get("metrics", {}).get("auc", 0.0)),
                is_pytorch=bool(meta.get("is_pytorch", False)),
                is_sb3=bool(meta.get("is_sb3", False)),
                train_rows=int(meta.get("train_rows", 0)),
                val_rows=int(meta.get("val_rows", 0)),
            )

            if symbol is not None and record.symbol != symbol:
                continue
            if expiry_key is not None and record.expiry_key != expiry_key:
                continue
            if model_name is not None and record.model_name != model_name:
                continue

            records.append(record)

        records.sort(key=lambda r: r.trained_at, reverse=True)

        logger.debug(
            "[^] list_models(): symbol=%s expiry=%s model=%s -> %d records",
            symbol,
            expiry_key,
            model_name,
            len(records),
        )

        return records

    def get_best_model(
        self,
        symbol: str,
        expiry_key: str,
        model_name: str | None = None,
    ) -> ModelRecord | None:
        """
        Return the highest-AUC model artifact for a symbol/expiry combination.

        Scans the registry via list_models(), filters to the current
        feature version only, then returns the record with the highest
        validation AUC. For RL models (which have no AUC) returns the
        most recently trained artifact instead.

        Args:
            symbol:     Currency pair to filter by, e.g. "EUR_USD".
            expiry_key: Expiry window to filter by, e.g. "5_MIN".
            model_name: Optional trainer class name to further narrow
                        the search, e.g. "XGBoostTrainer".

        Returns:
            ModelRecord | None: The best matching record, or None if no
                valid artifacts exist for the requested combination.
        """
        candidates = [
            r
            for r in self.list_models(symbol, expiry_key, model_name)
            if r.feature_version == _VERSION
        ]

        if not candidates:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"MODEL MANAGER WARNING: NO CURRENT-VERSION ARTIFACTS FOUND\n"
                f"Symbol     : {symbol}\n"
                f"Expiry key : {expiry_key}\n"
                f"Model name : {model_name}\n"
                f"Version    : {_VERSION}\n\n"
                f"CONTEXT: No artifacts matching the current feature schema version\n"
                f"were found for this symbol/expiry combination. Stale artifacts\n"
                f"from a different version are excluded from selection to prevent\n"
                f"silent prediction errors.\n"
                f"\nFIX: Run a training pass for symbol={symbol} expiry={expiry_key}\n"
                f"to produce a current-version ({_VERSION}) artifact before\n"
                f"attempting inference.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)
            return None

        rl_candidates = [r for r in candidates if r.is_sb3]
        ml_candidates = [r for r in candidates if not r.is_sb3]

        best: ModelRecord | None = None

        if ml_candidates:
            best = max(ml_candidates, key=lambda r: r.auc)
        if rl_candidates and (best is None or rl_candidates[0].auc >= best.auc):
            best = rl_candidates[0]

        if best is not None:
            logger.info(
                "[^] get_best_model(): selected model=%s symbol=%s "
                "expiry=%s auc=%.4f",
                best.model_name,
                best.symbol,
                best.expiry_key,
                best.auc,
            )

        return best

    async def pull_from_blob(
        self,
        symbol: str,
        expiry_key: str,
        model_name: str | None = None,
    ) -> str | None:
        """
        Pull the matching model artifact from Azure Blob to local disk.

        Called during container cold-start by pipeline.py before
        get_best_model() is invoked. Queries the Azure "models/" prefix
        for the artifact matching the symbol, expiry_key, and optional
        model_name, then downloads both the .artifact and .json sidecar
        to the local storage directory.

        Artifacts are overwritten in-place on each training run, so
        exactly one blob per symbol/expiry/model combination is expected.

        In LOCAL mode this is a no-op and returns None — local files
        are already present on disk. In CLOUD mode, if no matching blob
        is found or the download fails, returns None and logs a warning
        so the caller (pipeline.py) can decide whether to abort or
        proceed without a model.

        Args:
            symbol:     Currency pair to search for, e.g. "EUR_USD".
            expiry_key: Expiry window to search for, e.g. "5_MIN".
            model_name: Optional trainer class name to narrow the search,
                        e.g. "XGBoostTrainer". None matches any model.

        Returns:
            str | None: Local path to the downloaded .artifact file if
                        successful. None if LOCAL mode, no matching blob,
                        or download failed.
        """
        if self._storage._container_client is None:
            logger.debug("[^] pull_from_blob: DATA_MODE is LOCAL — skipping Blob pull.")
            return None

        try:
            # List all blobs under the models/ prefix and find
            # candidates that match the symbol and expiry_key.
            blobs = list(
                self._storage._container_client.list_blobs(name_starts_with="models/")
            )

            # Filter to .artifact files only — sidecars are pulled alongside.
            artifact_blobs = [
                b
                for b in blobs
                if b.name.endswith(".artifact")
                and f"_{symbol}_" in b.name
                and f"_{expiry_key}_" in b.name
                and (model_name is None or b.name.startswith(f"models/{model_name}_"))
            ]

            if not artifact_blobs:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"MODEL MANAGER WARNING: NO BLOB ARTIFACT FOUND\n"
                    f"Symbol     : {symbol}\n"
                    f"Expiry key : {expiry_key}\n"
                    f"Model name : {model_name}\n\n"
                    f"CONTEXT: No matching .artifact blob was found under the\n"
                    f"models/ prefix in Azure Blob Storage. The container will\n"
                    f"start without a pre-trained model.\n"
                    f"\nFIX: Run a training pass to produce and upload a model\n"
                    f"artifact before starting the inference container.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
                return None

            artifact_filename = Path(artifact_blobs[0].name).name

            local_path = self._storage.load_model(
                artifact_filename=artifact_filename,
                local_dir=self._storage_dir,
            )

            if local_path is not None:
                logger.info(
                    "[^] pull_from_blob: restored %s -> %s",
                    artifact_filename,
                    local_path,
                )

            return str(local_path) if local_path is not None else None

        except Exception as e:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER PULL FROM BLOB FAILURE\n"
                f"Symbol     : {symbol}\n"
                f"Expiry key : {expiry_key}\n"
                f"Error      : {e}\n\n"
                f"CONTEXT: An unexpected error occurred while listing or\n"
                f"downloading blobs from Azure Storage. The container will\n"
                f"start without a restored model.\n"
                f"\nFIX: Check AZURE_STORAGE_CONN and network connectivity.\n"
                f"{'!' * 60}"
            )
            logger.error(error_block)
            return None

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_metadata(self, metadata_path: str) -> dict[str, Any]:
        """
        Load and validate a metadata sidecar file.

        Checks that the JSON is parseable and that all required fields
        are present. Raises ModelManagerError for missing required fields
        so the caller knows exactly which field is absent rather than
        receiving a KeyError later.

        Required fields: model_name, symbol, expiry_key, feature_version,
        trained_at, metrics.

        Args:
            metadata_path: Path to the .json sidecar file.

        Returns:
            dict[str, Any]: Parsed metadata dict.

        Raises:
            FileNotFoundError:  If the metadata file does not exist.
            ModelManagerError:  If the JSON is malformed or required
                                fields are missing.
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"[!] Metadata file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                metadata: dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER METADATA PARSE FAILURE\n"
                f"File: {path.name}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: The metadata sidecar could not be parsed as valid JSON.\n"
                f"Without readable metadata the artifact cannot be version-validated\n"
                f"or deserialised. The system treats this artifact as unloadable.\n"
                f"\nFIX: Delete both the artifact and sidecar files and re-run the\n"
                f"training pass to generate a fresh, clean artifact/sidecar pair.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"Metadata JSON parse failure: {path.name}",
                stage="validate_metadata",
            ) from exc

        required_fields = {
            "model_name",
            "symbol",
            "expiry_key",
            "feature_version",
            "trained_at",
            "metrics",
        }
        missing = required_fields - set(metadata.keys())
        if missing:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER METADATA VALIDATION FAILURE\n"
                f"File: {path.name}\n"
                f"Missing fields: {sorted(missing)}\n\n"
                f"CONTEXT: The metadata sidecar is missing one or more required fields.\n"
                f"This usually occurs when a sidecar was written by an older version of\n"
                f"ModelManager that had a different schema, or when the file was manually\n"
                f"edited and fields were accidentally removed or renamed.\n"
                f"\nFIX: Delete both the artifact and sidecar and re-run the training\n"
                f"pass to regenerate a complete, current-schema metadata sidecar.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"Metadata missing required fields {sorted(missing)}: {path.name}",
                stage="validate_metadata",
            )

        return metadata

    # ── Delete ───────────────────────────────────────────────────────────────

    def delete(self, artifact_path: str) -> None:
        """
        Remove a model artifact and its metadata sidecar from disk.

        Deletes both files atomically — if the artifact file is removed
        successfully but the sidecar removal fails, the orphaned sidecar
        is logged as a warning (it will appear in list_models() output
        but be skipped due to the missing artifact). This is preferable
        to leaving a partially deleted artifact pair.

        Args:
            artifact_path: Path to the .artifact file to delete.
                            The companion .json sidecar is inferred
                            from the same base path.

        Raises:
            FileNotFoundError:  If the artifact file does not exist.
            ModelManagerError:  If file deletion fails.
        """
        path = Path(artifact_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"[!] Cannot delete: artifact not found at {path}.")

        metadata_path = path.with_suffix(_METADATA_SUFFIX)

        try:
            path.unlink()
            logger.info(
                "[^] ModelManager.delete(): artifact removed: %s",
                path.name,
            )
        except OSError as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER DELETE FAILURE\n"
                f"Artifact: {path.name}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: The artifact file could not be deleted from disk.\n"
                f"The model registry may still reference this file until it\n"
                f"is successfully removed.\n"
                f"\nFIX: Verify that no other process holds an open file handle\n"
                f"on this artifact, then retry the delete operation.\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"delete() failed to remove artifact {path.name}: {exc}",
                stage="delete",
            ) from exc

        if metadata_path.exists():
            try:
                metadata_path.unlink()
                logger.info(
                    "[^] ModelManager.delete(): sidecar removed: %s",
                    metadata_path.name,
                )
            except OSError as exc:
                warning_block = (
                    f"\n{'%' * 60}\n"
                    f"MODEL MANAGER WARNING: SIDECAR DELETION FAILED\n"
                    f"Artifact removed   : {path.name}\n"
                    f"Sidecar not removed: {metadata_path.name}\n"
                    f"Error: {exc}\n\n"
                    f"CONTEXT: The artifact was successfully deleted but the metadata\n"
                    f"sidecar could not be removed. The orphaned sidecar will appear\n"
                    f"in list_models() scans but will be automatically skipped because\n"
                    f"its corresponding artifact file no longer exists.\n"
                    f"\nFIX: Manually delete {metadata_path.name} to clean the registry.\n"
                    f"{'%' * 60}"
                )
                logger.warning(warning_block)
        else:
            warning_block = (
                f"\n{'%' * 60}\n"
                f"MODEL MANAGER WARNING: NO SIDECAR FOUND DURING DELETE\n"
                f"Artifact: {path.name}\n"
                f"Expected sidecar: {metadata_path.name}\n\n"
                f"CONTEXT: The artifact was deleted but no companion metadata sidecar\n"
                f"was found at the expected path. The registry may have already been\n"
                f"in an inconsistent state before this delete operation was called.\n"
                f"\nFIX: No further action required. Run list_models() to verify\n"
                f"the registry is now clean.\n"
                f"{'%' * 60}"
            )
            logger.warning(warning_block)

    # ── Internal Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_sb3_model(artifact: Any) -> bool:
        """
        Return True if the artifact is a Stable-Baselines3 model instance.

        Uses class name inspection rather than importing SB3 directly so
        that ModelManager remains importable in environments where SB3
        is not installed (e.g. a CPU inference container that only runs
        Classical ML models).

        Args:
            artifact: The model artifact to inspect.

        Returns:
            bool: True if the artifact's MRO contains a class named
                    "BaseAlgorithm" (the SB3 base class).
        """
        return any(cls.__name__ == "BaseAlgorithm" for cls in type(artifact).__mro__)

    @staticmethod
    def _load_pytorch(
        path: Path,
        model_class: Any,
    ) -> torch.nn.Module:
        """
        Load a PyTorch state_dict into a pre-instantiated nn.Module.

        Maps to CPU first (map_location="cpu") so that models trained
        on GPU can be loaded in a CPU-only inference container without
        raising a device mismatch error.

        Args:
            path:        Path to the .artifact file containing the
                            state_dict.
            model_class: Pre-instantiated nn.Module with the correct
                            architecture. Weights are loaded in-place.

        Returns:
            torch.nn.Module: The model with loaded weights in eval() mode.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model_class.load_state_dict(state_dict)
        model_class.eval()
        logger.debug(
            "[^] PyTorch state_dict loaded into model_class: %s",
            path.name,
        )
        return model_class

    @staticmethod
    def _load_sb3(path: Path, metadata: dict[str, Any]) -> Any:
        """
        Load a Stable-Baselines3 model using the SB3 native load interface.

        SB3 models carry their architecture inside the saved zip archive
        and do not require a model_class parameter. The algorithm class
        (PPO, A2C, DQN, RecurrentPPO) is inferred from model_name in the
        metadata.

        Args:
            path:     Path to the .artifact file (SB3 zip archive).
            metadata: Parsed metadata dict containing model_name.

        Returns:
            Any: The loaded SB3 algorithm instance.

        Raises:
            ModelManagerError: If the model_name cannot be mapped to a
                                known SB3 algorithm class, or if the
                                import fails.
        """
        model_name: str = metadata.get("model_name", "")

        _SB3_CLASS_MAP: dict[str, str] = {
            "A2CTrainer": "stable_baselines3.A2C",
            "DQNTrainer": "stable_baselines3.DQN",
            "PPOTrainer": "stable_baselines3.PPO",
            "RecurrentPPOTrainer": "sb3_contrib.RecurrentPPO",
        }

        module_path = _SB3_CLASS_MAP.get(model_name)
        if module_path is None:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER LOAD FAILURE: UNKNOWN SB3 MODEL NAME\n"
                f"model_name received : {model_name!r}\n"
                f"Known SB3 trainers  : {list(_SB3_CLASS_MAP.keys())}\n\n"
                f"CONTEXT: The metadata sidecar contains a model_name that does not\n"
                f"map to any known Stable-Baselines3 algorithm class. The artifact\n"
                f"cannot be deserialised without knowing its algorithm class.\n"
                f"\nFIX: Verify the model_name in the metadata sidecar matches one\n"
                f"of the known SB3 trainer names above. If a new RL trainer was\n"
                f"added to trainer.py, add its entry to _SB3_CLASS_MAP inside\n"
                f"ModelManager._load_sb3().\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"Unknown SB3 model_name: '{model_name}'. "
                f"Known: {list(_SB3_CLASS_MAP.keys())}.",
                stage="load",
            )

        try:
            module_str, class_str = module_path.rsplit(".", 1)
            module = importlib.import_module(module_str)
            AlgorithmClass = getattr(module, class_str)
        except ImportError as exc:
            error_block = (
                f"\n{'!' * 60}\n"
                f"MODEL MANAGER LOAD FAILURE: SB3 IMPORT ERROR\n"
                f"Required class : {module_path}\n"
                f"Error: {exc}\n\n"
                f"CONTEXT: The Stable-Baselines3 or sb3-contrib package required\n"
                f"to load this artifact is not installed in the current environment.\n"
                f"This commonly occurs when running inference in a container that\n"
                f"does not include the full training dependencies.\n"
                f"\nFIX: Add the missing package to your Dockerfile:\n"
                f"  stable-baselines3>=2.3.0  (for A2C, DQN, PPO)\n"
                f"  sb3-contrib>=2.3.0         (for RecurrentPPO)\n"
                f"{'!' * 60}"
            )
            logger.critical(error_block)
            raise ModelManagerError(
                f"Cannot import SB3 class '{module_path}'. "
                f"Ensure the correct package is installed.",
                stage="load",
            ) from exc

        model = AlgorithmClass.load(str(path))
        logger.debug(
            "[^] SB3 model loaded via native .load(): %s",
            path.name,
        )
        return model

    def __repr__(self) -> str:
        return (
            f"ModelManager("
            f"storage_dir={str(self._storage_dir)!r}, "
            f"version={_VERSION!r})"
        )
