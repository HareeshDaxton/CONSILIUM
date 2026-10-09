"""Section 1.2 — vision contracts (AGENTS.md §7, §8)."""

import pytest
from pydantic import ValidationError

from consilium.schemas.enums import Laterality, QualityFlag, ScreeningTier
from consilium.schemas.vision import (
    CADResult,
    CDRMetrics,
    ClassifierOutput,
    ImageQuality,
    MaskQC,
    Measurements,
    RawVisionOutput,
)


def _quality() -> ImageQuality:
    return ImageQuality(score=0.9, gradable=True, flags=())


def _classifier() -> ClassifierOutput:
    return ClassifierOutput(
        p_raw=0.71,
        p_calibrated=0.66,
        tier=ScreeningTier.INTERMEDIATE,
        model_version="swinv2@rev-abc123+mlflow-run-1",
    )


def _measurements() -> Measurements:
    return Measurements(
        disc_area_px=12000,
        cup_area_px=4800,
        disc_vertical_diameter_px=200,
        cup_vertical_diameter_px=124,
    )


def _mask_qc() -> MaskQC:
    return MaskQC(
        cup_within_disc=True,
        single_component_each=True,
        disc_area_plausible=True,
        ok=True,
    )


def _cad_result() -> CADResult:
    return CADResult(
        quality=_quality(),
        classifier=_classifier(),
        measurements=_measurements(),
        cdr=CDRMetrics(vertical=0.62, area_based=0.63),
        mask_qc=_mask_qc(),
        segmenter_version="segformer@rev-def456+mlflow-run-2",
        laterality=Laterality.OD,
    )


class TestImageQuality:
    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ImageQuality(score=1.1, gradable=True)
        with pytest.raises(ValidationError):
            ImageQuality(score=-0.1, gradable=True)

    def test_flags_default_empty(self) -> None:
        assert _quality().flags == ()

    def test_flags_typed(self) -> None:
        q = ImageQuality(score=0.4, gradable=False, flags=(QualityFlag.UNDEREXPOSED,))
        assert q.flags == (QualityFlag.UNDEREXPOSED,)


class TestClassifierOutput:
    @pytest.mark.parametrize("field", ["p_raw", "p_calibrated"])
    def test_probability_bounds(self, field: str) -> None:
        kwargs: dict[str, object] = {
            "p_raw": 0.5,
            "p_calibrated": 0.5,
            "tier": ScreeningTier.LOW,
            "model_version": "v",
            field: 1.01,
        }
        with pytest.raises(ValidationError):
            ClassifierOutput(**kwargs)  # type: ignore[arg-type]


class TestMaskQC:
    def test_ok_is_stored_as_given(self) -> None:
        """`ok` is a stored AND computed by vision code, not derived here."""
        qc = MaskQC(
            cup_within_disc=True,
            single_component_each=False,
            disc_area_plausible=True,
            ok=False,
        )
        assert qc.ok is False


class TestMeasurements:
    def test_disc_area_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Measurements(
                disc_area_px=0,
                cup_area_px=0,
                disc_vertical_diameter_px=10,
                cup_vertical_diameter_px=0,
            )

    def test_cup_may_be_zero(self) -> None:
        m = Measurements(
            disc_area_px=100,
            cup_area_px=0,
            disc_vertical_diameter_px=10,
            cup_vertical_diameter_px=0,
        )
        assert m.cup_area_px == 0

    def test_rejects_float_pixels(self) -> None:
        with pytest.raises(ValidationError):
            Measurements(
                disc_area_px=100.5,  # type: ignore[arg-type]
                cup_area_px=0,
                disc_vertical_diameter_px=10,
                cup_vertical_diameter_px=0,
            )


class TestCDRMetrics:
    def test_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CDRMetrics(vertical=1.2, area_based=0.5)

    def test_mask_convention_default(self) -> None:
        assert CDRMetrics(vertical=0.5, area_based=0.5).mask_convention == "disc_includes_cup"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CDRMetrics(vertical=0.5, area_based=0.5, bogus=1)  # type: ignore[call-arg]


class TestCADResult:
    def test_round_trip_json(self) -> None:
        result = _cad_result()
        restored = CADResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_frozen(self) -> None:
        result = _cad_result()
        with pytest.raises(ValidationError):
            result.laterality = Laterality.OS  # type: ignore[misc]

    def test_dump_shape(self) -> None:
        dumped = _cad_result().model_dump(mode="json")
        assert dumped["classifier"]["tier"] == "intermediate"
        assert dumped["laterality"] == "OD"
        assert dumped["cdr"]["mask_convention"] == "disc_includes_cup"


class TestRawVisionOutput:
    def _raw(self) -> RawVisionOutput:
        return RawVisionOutput(
            p_raw=0.71,
            disc_mask_rle="0 100 1 50",
            cup_mask_rle="120 20",
            mask_width=512,
            mask_height=512,
            endpoint_revision="rev-abc123",
        )

    def test_valid(self) -> None:
        raw = self._raw()
        assert raw.mask_width == 512

    def test_mask_dims_positive(self) -> None:
        with pytest.raises(ValidationError):
            RawVisionOutput(
                p_raw=0.5,
                disc_mask_rle="x",
                cup_mask_rle="y",
                mask_width=0,
                mask_height=512,
                endpoint_revision="rev",
            )

    def test_round_trip(self) -> None:
        raw = self._raw()
        assert RawVisionOutput.model_validate_json(raw.model_dump_json()) == raw

    def test_revision_pin_not_checked_in_schema(self) -> None:
        """Any revision string parses; the VisionClient enforces the pin (I-13)."""
        raw = self._raw()
        assert raw.endpoint_revision == "rev-abc123"
