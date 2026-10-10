"""SegFormer disc/cup segmentation fine-tuning (P2, Section 2.2).

3 classes under the nested convention: 0 background, 1 disc rim, 2 cup
(disc mask INCLUDES cup). Metrics per AGENTS.md §8.3: Dice + IoU for disc
and cup, vCDR MAE vs expert annotation. Config-driven, seeded, MLflow-logged
— same provenance contract as classification (git SHA, seed, data version).

Model policy (§8.1): nvidia/mit-b0 (smallest SegFormer) first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from training.common.runs import current_git_sha
from training.common.seed import seed_all
from training.segmentation.metrics import dice_score, iou_score, vcdr_mae

NUM_CLASSES = 3


@dataclass(frozen=True)
class SegModelConfig:
    pretrained: bool = True
    model_id: str = "nvidia/mit-b0"
    tiny: bool = False  # offline smoke only


@dataclass(frozen=True)
class SegHyperparams:
    epochs: int = 40
    batch_size: int = 4
    lr: float = 6e-5
    weight_decay: float = 0.01
    seed: int = 42
    num_workers: int = 0


@dataclass(frozen=True)
class SegTrainConfig:
    model: SegModelConfig = field(default_factory=SegModelConfig)
    training: SegHyperparams = field(default_factory=SegHyperparams)
    mlflow_tracking_uri: str = "sqlite:///./mlruns/mlflow.db"
    mlflow_experiment: str = "consilium-segmentation"
    data_version: str = "unknown"


def load_config(path: Path) -> SegTrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SegTrainConfig(
        model=SegModelConfig(**raw.get("model", {})),
        training=SegHyperparams(**raw.get("training", {})),
        mlflow_tracking_uri=raw.get("mlflow", {}).get(
            "tracking_uri", "sqlite:///./mlruns/mlflow.db"
        ),
        mlflow_experiment=raw.get("mlflow", {}).get("experiment", "consilium-segmentation"),
        data_version=raw.get("data", {}).get("metadata_hash", "unknown"),
    )


def build_model(cfg: SegModelConfig) -> torch.nn.Module:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if cfg.tiny:
        tiny = SegformerConfig(
            num_encoder_blocks=2,
            depths=[1, 1],
            hidden_sizes=[16, 32],
            decoder_hidden_size=32,
            num_labels=NUM_CLASSES,
            image_size=64,
            patch_sizes=[4, 2],
            strides=[4, 2],
            num_attention_heads=[1, 2],
            mlp_ratios=[2, 2],
            sr_ratios=[1, 1],
        )
        return SegformerForSemanticSegmentation(tiny)
    return SegformerForSemanticSegmentation.from_pretrained(
        cfg.model_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], device: str
) -> dict[str, float]:
    model.eval()
    disc_dice: list[float] = []
    cup_dice: list[float] = []
    disc_iou: list[float] = []
    cup_iou: list[float] = []
    vcdr_err: list[float] = []
    for batch in loader:
        logits = model(pixel_values=batch["pixel_values"].to(device)).logits
        upsampled = F.interpolate(
            logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).cpu().numpy()
        target = batch["labels"].numpy()
        for p, t in zip(pred, target, strict=True):
            pred_disc, pred_cup = p >= 1, p == 2
            gt_disc, gt_cup = t >= 1, t == 2
            disc_dice.append(dice_score(pred_disc, gt_disc))
            cup_dice.append(dice_score(pred_cup, gt_cup))
            disc_iou.append(iou_score(pred_disc, gt_disc))
            cup_iou.append(iou_score(pred_cup, gt_cup))
            if gt_disc.any() and pred_disc.any():
                vcdr_err.append(vcdr_mae(pred_disc, pred_cup, gt_disc, gt_cup))
    return {
        "disc_dice": float(np.mean(disc_dice)),
        "cup_dice": float(np.mean(cup_dice)),
        "disc_iou": float(np.mean(disc_iou)),
        "cup_iou": float(np.mean(cup_iou)),
        "vcdr_mae": float(np.mean(vcdr_err)) if vcdr_err else float("nan"),
    }


def train(
    cfg: SegTrainConfig,
    train_ds: Dataset[dict[str, torch.Tensor]],
    val_ds: Dataset[dict[str, torch.Tensor]],
    *,
    run_name: str = "segformer-refuge",
) -> dict[str, float]:
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
                    labels=batch["labels"].to(device),
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += float(outputs.loss.detach())
            metrics = evaluate_split(model, val_loader, device)
            metrics["train_loss"] = total_loss / max(1, len(loader))
            mlflow.log_metrics(metrics, step=epoch)
            final_metrics = metrics
    return final_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="segformer-refuge")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))["data"]

    from training.common.dataset import FundusSegmentationDataset
    from training.common.preprocessing import load_spec
    from training.data.manifest import load_manifest
    from training.data.split import SplitManifest

    metadata = load_manifest(Path(raw["metadata_manifest"]))
    split = SplitManifest.model_validate_json(Path(raw["split_manifest"]).read_text())
    spec = load_spec()
    images_root = Path(raw["images_root"])
    train_ds = FundusSegmentationDataset(metadata, split, "train", images_root, spec)
    val_ds = FundusSegmentationDataset(metadata, split, "val", images_root, spec)
    cfg = SegTrainConfig(
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
