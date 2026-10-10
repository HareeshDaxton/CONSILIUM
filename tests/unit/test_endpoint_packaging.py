"""Section 2.4 — endpoint packaging + handler parity."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from training.endpoint_packaging.export import build_revision, export_package

pytest.importorskip("torch", reason="training group not installed")

from training.common.preprocessing import PreprocessingSpec


class TestRevision:
    def test_deterministic(self) -> None:
        kwargs = {
            "classifier_sha256": "a" * 64,
            "segmenter_sha256": "b" * 64,
            "spec_sha256": "c" * 64,
            "classifier_run_id": "run-1",
            "segmenter_run_id": "run-2",
        }
        assert build_revision(**kwargs) == build_revision(**kwargs)

    def test_changes_with_content(self) -> None:
        base = {
            "classifier_sha256": "a" * 64,
            "segmenter_sha256": "b" * 64,
            "spec_sha256": "c" * 64,
            "classifier_run_id": "run-1",
            "segmenter_run_id": "run-2",
        }
        changed = build_revision(**{**base, "classifier_run_id": "run-9"})
        assert changed != build_revision(**base)


class TestExportPackage:
    def test_export_layout_and_manifest(self, tmp_path: Path) -> None:
        clf = tmp_path / "clf"
        seg = tmp_path / "seg"
        clf.mkdir()
        seg.mkdir()
        (clf / "model.safetensors").write_bytes(b"fake-weights-1")
        (seg / "model.safetensors").write_bytes(b"fake-weights-2")
        spec = tmp_path / "spec.json"
        spec.write_text(PreprocessingSpec().to_json(), encoding="utf-8")

        package = export_package(
            classifier_dir=clf,
            segmenter_dir=seg,
            preprocessing_spec=spec,
            out_dir=tmp_path / "endpoint",
            classifier_run_id="mlflow-cls-1",
            segmenter_run_id="mlflow-seg-1",
            classifier_model_id="swinv2-small",
            segmenter_model_id="mit-b0",
        )

        out = package.out_dir
        assert (out / "classifier" / "model.safetensors").exists()
        assert (out / "segmenter" / "model.safetensors").exists()
        assert (out / "preprocessing_spec.json").exists()
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["revision"] == package.revision
        assert package.revision.startswith("consilium-vision-")
        assert manifest["classifier"]["mlflow_run_id"] == "mlflow-cls-1"

    def test_export_deterministic_revision(self, tmp_path: Path) -> None:
        def make(root: Path) -> str:
            clf, seg = root / "clf", root / "seg"
            clf.mkdir(parents=True)
            seg.mkdir(parents=True)
            (clf / "w.bin").write_bytes(b"w1")
            (seg / "w.bin").write_bytes(b"w2")
            spec = root / "spec.json"
            spec.write_text(PreprocessingSpec().to_json(), encoding="utf-8")
            return export_package(
                classifier_dir=clf,
                segmenter_dir=seg,
                preprocessing_spec=spec,
                out_dir=root / "out",
                classifier_run_id="r1",
                segmenter_run_id="r2",
                classifier_model_id="m1",
                segmenter_model_id="m2",
            ).revision

        assert make(tmp_path / "a") == make(tmp_path / "b")


class TestHandlerParity:
    def test_preprocess_matches_training_path(self) -> None:
        """The endpoint's preprocessing must be pixel-identical to training's
        (train/serve skew control, AGENTS.md §8.0)."""
        from deploy.vision_endpoint.handler import preprocess_image
        from training.common.dataset import _load_image_tensor

        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 255, (128, 96, 3), dtype=np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        spec = PreprocessingSpec()
        handler_side = preprocess_image(image_bytes, json.loads(spec.to_json()))

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(image_bytes)
            tmp_image = Path(fh.name)
        training_tensor = _load_image_tensor(tmp_image, spec).numpy()
        tmp_image.unlink()

        assert handler_side.shape == training_tensor.shape
        np.testing.assert_allclose(handler_side, training_tensor, atol=1e-6)

    def test_rle_encode_matches_serving(self) -> None:
        """The handler's RLE must byte-match the serving decoder's format."""
        from deploy.vision_endpoint.handler import encode_rle as handler_encode

        from consilium.vision.rle import decode_rle
        from consilium.vision.rle import encode_rle as serving_encode

        rng = np.random.default_rng(1)
        mask = rng.random((61, 47)) > 0.5
        assert handler_encode(mask) == serving_encode(mask)
        assert np.array_equal(decode_rle(handler_encode(mask), 61, 47), mask)
