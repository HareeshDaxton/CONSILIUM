"""Section 2.2 — dataset loading + offline training smoke (tiny random-init
models, synthetic tensors, no network, no pretrained downloads)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training group not installed")
pytest.importorskip("transformers", reason="training group not installed")
pytest.importorskip("mlflow", reason="training group not installed")

from torch.utils.data import Dataset  # noqa: E402
from training.classification.train import (  # noqa: E402
    ClsTrainConfig,
    ModelConfig,
    TrainHyperparams,
)
from training.classification.train import build_model as build_cls_model  # noqa: E402
from training.classification.train import train as train_cls  # noqa: E402
from training.common.preprocessing import PreprocessingSpec  # noqa: E402
from training.data.manifest import ImageRecord, MetadataManifest  # noqa: E402
from training.data.split import split_by_patient  # noqa: E402
from training.segmentation.train import SegHyperparams, SegModelConfig, SegTrainConfig  # noqa: E402
from training.segmentation.train import build_model as build_seg_model  # noqa: E402
from training.segmentation.train import train as train_seg  # noqa: E402

SPEC = PreprocessingSpec(image_size=64)
IMAGE_SIZE = 64


class _SyntheticCls(Dataset):
    def __init__(self, n: int = 4) -> None:
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(n, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        self._x = torch.from_numpy(noise)
        self._y = torch.tensor([0, 1] * (n // 2), dtype=torch.long)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"pixel_values": self._x[i], "label": self._y[i]}


class _SyntheticSeg(Dataset):
    def __init__(self, n: int = 4) -> None:
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(n, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        self._x = torch.from_numpy(noise)
        yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
        dist = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
        labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        labels[dist <= 20] = 1
        labels[dist <= 8] = 2
        self._m = torch.from_numpy(np.stack([labels] * n))

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"pixel_values": self._x[i], "labels": self._m[i]}


def test_tiny_classifier_forward_backward() -> None:
    model = build_cls_model(ModelConfig(pretrained=False, tiny=True))
    batch = _SyntheticCls()[0]
    out = model(pixel_values=batch["pixel_values"].unsqueeze(0), labels=batch["label"].unsqueeze(0))
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_tiny_segmenter_forward_backward() -> None:
    model = build_seg_model(SegModelConfig(pretrained=False, tiny=True))
    batch = _SyntheticSeg()[0]
    out = model(
        pixel_values=batch["pixel_values"].unsqueeze(0),
        labels=batch["labels"].unsqueeze(0),
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_classification_train_run_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    # MLflow 3 deprecated the file store (requires opt-out). Tests use it
    # because the sqlite backend needs SQLAlchemy, whose C extension is
    # blocked by some Windows Application Control policies; file store works
    # identically on every host for this offline smoke.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = ClsTrainConfig(
        model=ModelConfig(pretrained=False, tiny=True),
        training=TrainHyperparams(epochs=1, batch_size=2, seed=123),
        mlflow_tracking_uri=f"file:{tmp_path}/mlruns",
        mlflow_experiment="smoke-cls",
        data_version="synthetic-v0",
    )
    metrics = train_cls(cfg, _SyntheticCls(), _SyntheticCls(), run_name="smoke")

    assert set(metrics) >= {"auroc", "brier", "ece", "train_loss"}
    client = mlflow.tracking.MlflowClient(f"file:{tmp_path}/mlruns")
    experiment = client.get_experiment_by_name("smoke-cls")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    params = runs[0].data.params
    assert params["seed"] == "123"
    assert params["data_version"] == "synthetic-v0"
    assert "git_sha" in params
    assert "auroc" in runs[0].data.metrics


def test_segmentation_train_run_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # see classification smoke
    cfg = SegTrainConfig(
        model=SegModelConfig(pretrained=False, tiny=True),
        training=SegHyperparams(epochs=1, batch_size=2, seed=123),
        mlflow_tracking_uri=f"file:{tmp_path}/mlruns",
        mlflow_experiment="smoke-seg",
        data_version="synthetic-v0",
    )
    metrics = train_seg(cfg, _SyntheticSeg(), _SyntheticSeg(), run_name="smoke")

    assert set(metrics) >= {"disc_dice", "cup_dice", "disc_iou", "cup_iou", "train_loss"}
    client = mlflow.tracking.MlflowClient(f"file:{tmp_path}/mlruns")
    experiment = client.get_experiment_by_name("smoke-seg")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["data_version"] == "synthetic-v0"
    assert "disc_dice" in runs[0].data.metrics


class _DiskFixtures:
    def __init__(self, root: Path) -> None:
        from PIL import Image

        images = root / "images"
        masks = root / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        rng = np.random.default_rng(1)
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(images / "img1.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        yy, xx = np.mgrid[0:64, 0:64]
        dist = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
        mask[dist <= 20] = 255  # disc (includes cup)
        mask[dist <= 8] = 128  # cup
        Image.fromarray(mask).save(masks / "img1.png")
        self.metadata = MetadataManifest(
            dataset="T",
            records=(
                ImageRecord(
                    image_id="img1",
                    patient_id="p1",
                    label=1,
                    mask_path=str(masks / "img1.png"),
                ),
            ),
        )
        self.split = split_by_patient(self.metadata, seed=1, ratios={"train": 1.0}, name="t")
        self.images = images


def test_classification_dataset_from_disk(tmp_path: Path) -> None:
    from training.common.dataset import FundusClassificationDataset

    fx = _DiskFixtures(tmp_path)
    ds = FundusClassificationDataset(fx.metadata, fx.split, "train", fx.images, SPEC)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 64, 64)
    assert sample["label"].item() == 1


def test_segmentation_dataset_mask_convention(tmp_path: Path) -> None:
    from training.common.dataset import FundusSegmentationDataset

    fx = _DiskFixtures(tmp_path)
    ds = FundusSegmentationDataset(fx.metadata, fx.split, "train", fx.images, SPEC)
    sample = ds[0]
    labels = sample["labels"]
    assert labels.shape == (64, 64)
    assert set(labels.unique().tolist()) == {0, 1, 2}
    # nested convention: disc region includes cup; rim pixels are class 1
    assert (labels == 2).sum() > 0
    assert (labels == 1).sum() > (labels == 2).sum()
