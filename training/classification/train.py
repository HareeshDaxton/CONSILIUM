"""SwinV2 binary glaucoma classification fine-tuning (P2, Section 2.2).

Config-driven (YAML), seeded, MLflow-logged (AGENTS.md §8.1, §19). Every run
records git SHA, seed, config, and the data version (metadata content hash).

Model policy (§8.1): start with a SMALL variant (swinv2-small); any larger
backbone needs a measured gain in a stored MLflow run. `tiny: true` builds a
randomly-initialized miniature from config — offline smoke tests only, never
for reported metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from training.classification.metrics import (
    auroc,
    brier_score,
    expected_calibration_error,
    operating_point,
    reliability_curve,
)
from training.common.runs import current_git_sha
from training.common.seed import seed_all


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool = True
    model_id: str = "microsoft/swinv2-small-patch4-window8-256"
    tiny: bool = False  # offline smoke only


@dataclass(frozen=True)
class TrainHyperparams:
    epochs: int = 20
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    seed: int = 42
    num_workers: int = 0


@dataclass(frozen=True)
class ClsTrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainHyperparams = field(default_factory=TrainHyperparams)
    mlflow_tracking_uri: str = "sqlite:///./mlruns/mlflow.db"
    mlflow_experiment: str = "consilium-classification"
    data_version: str = "unknown"  # metadata content hash, filled by CLI


def load_config(path: Path) -> ClsTrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ClsTrainConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainHyperparams(**raw.get("training", {})),
        mlflow_tracking_uri=raw.get("mlflow", {}).get(
            "tracking_uri", "sqlite:///./mlruns/mlflow.db"
        ),
        mlflow_experiment=raw.get("mlflow", {}).get("experiment", "consilium-classification"),
        data_version=raw.get("data", {}).get("metadata_hash", "unknown"),
    )


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    from transformers import Swinv2Config, Swinv2ForImageClassification

    if cfg.tiny:
        tiny = Swinv2Config(
            image_size=64,
            patch_size=4,
            embed_dim=16,
            depths=[1, 1],
            num_heads=[2, 2],
            window_size=8,
            num_labels=2,
        )
        return Swinv2ForImageClassification(tiny)
    return Swinv2ForImageClassification.from_pretrained(
        cfg.model_id, num_labels=2, ignore_mismatched_sizes=True
    )


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        outputs = model(pixel_values=batch["pixel_values"].to(device))
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        scores.append(probs.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(scores)


def evaluate_split(
    model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], device: str
) -> dict[str, float]:
    y_true, y_score = predict_scores(model, loader, device)
    labeled = y_true >= 0
    y_true, y_score = y_true[labeled], y_score[labeled]
    metrics = {
        "auroc": auroc(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
    }
    sens_point = operating_point(y_true, y_score, target_specificity=0.95)
    spec_point = operating_point(y_true, y_score, target_sensitivity=0.95)
    metrics["sensitivity_at_spec95"] = sens_point.sensitivity
    metrics["specificity_at_sens95"] = spec_point.specificity
    return metrics


def train(
    cfg: ClsTrainConfig,
    train_ds: Dataset[dict[str, torch.Tensor]],
    val_ds: Dataset[dict[str, torch.Tensor]],
    *,
    run_name: str = "swinv2-refuge",
) -> dict[str, float]:
    """Full training run with MLflow logging. Returns final val metrics."""
    import mlflow

    hp = cfg.training
    seed_all(hp.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=hp.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, num_workers=hp.num_workers)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    final_metrics: dict[str, float] = {}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "git_sha": current_git_sha(),
                "seed": hp.seed,
                "data_version": cfg.data_version,
                "model_id": cfg.model.model_id,
                "pretrained": cfg.model.pretrained,
                "tiny": cfg.model.tiny,
                **{f"hp_{k}": v for k, v in asdict(hp).items()},
            }
        )
        for epoch in range(hp.epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    labels=batch["label"].to(device),
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += float(outputs.loss.detach())
            metrics = evaluate_split(model, val_loader, device)
            metrics["train_loss"] = total_loss / max(1, len(loader))
            mlflow.log_metrics(metrics, step=epoch)
            final_metrics = metrics

        y_true, y_score = predict_scores(model, val_loader, device)
        labeled = y_true >= 0
        artifact = reliability_curve(y_true[labeled], y_score[labeled])
        artifact_path = Path("reliability_curve.json")
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()
    return final_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="swinv2-refuge")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))["data"]

    from training.common.dataset import FundusClassificationDataset
    from training.common.preprocessing import load_spec
    from training.data.manifest import load_manifest
    from training.data.split import SplitManifest

    metadata = load_manifest(Path(raw["metadata_manifest"]))
    split = SplitManifest.model_validate_json(Path(raw["split_manifest"]).read_text())
    spec = load_spec()
    images_root = Path(raw["images_root"])
    train_ds = FundusClassificationDataset(metadata, split, "train", images_root, spec)
    val_ds = FundusClassificationDataset(metadata, split, "val", images_root, spec)
    cfg = ClsTrainConfig(
        model=cfg.model,
        training=cfg.training,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment=cfg.mlflow_experiment,
        data_version=metadata.content_hash(),
    )
    metrics = train(cfg, train_ds, val_ds, run_name=args.run_name)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
