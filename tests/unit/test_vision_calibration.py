"""Section 2.3 — calibration (AGENTS.md §8.3) and config loaders."""

import numpy as np
import pytest

from consilium.vision.calibration import calibrate, fit_temperature, logit, nll, sigmoid
from consilium.vision.config import load_models_config, load_thresholds


class TestCalibration:
    def test_logit_sigmoid_round_trip(self) -> None:
        assert sigmoid(logit(0.73)) == pytest.approx(0.73)

    def test_identity_at_temperature_one(self) -> None:
        assert calibrate(0.73, 1.0) == pytest.approx(0.73)

    def test_hot_temperature_pulls_toward_half(self) -> None:
        assert abs(calibrate(0.9, 3.0) - 0.5) < abs(0.9 - 0.5)
        assert calibrate(0.9, 3.0) > 0.5  # direction preserved

    def test_cold_temperature_sharpens(self) -> None:
        assert calibrate(0.6, 0.5) > 0.6

    def test_invalid_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            calibrate(0.5, 0.0)

    def test_fit_recovers_overconfidence(self) -> None:
        """Logits 3x overconfident → fitted T ≈ 3 and lower NLL than T=1."""
        rng = np.random.default_rng(0)
        z_true = rng.normal(0.0, 1.5, size=4000)
        labels = (rng.random(4000) < 1.0 / (1.0 + np.exp(-z_true))).astype(float)
        logits = 3.0 * z_true  # overconfident by a factor of 3
        t_fit = fit_temperature(logits, labels)
        assert t_fit == pytest.approx(3.0, abs=0.5)
        assert nll(logits, labels, t_fit) < nll(logits, labels, 1.0)

    def test_fit_well_calibrated_stays_near_one(self) -> None:
        rng = np.random.default_rng(1)
        z = rng.normal(0.0, 1.0, size=4000)
        labels = (rng.random(4000) < 1.0 / (1.0 + np.exp(-z))).astype(float)
        assert fit_temperature(z, labels) == pytest.approx(1.0, abs=0.25)

    def test_fit_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            fit_temperature(np.array([]), np.array([]))


class TestConfigLoaders:
    def test_thresholds_yaml_parses(self) -> None:
        cfg = load_thresholds()
        tiers = cfg.screening_tiers
        assert tiers.tier_for(0.1) == "low"
        assert tiers.tier_for(0.5) == "intermediate"
        assert tiers.tier_for(0.9) == "high"
        assert tiers.tier_for(0.70) == "high"  # boundary inclusive
        assert cfg.numeric_tolerances.cdr > 0

    def test_models_yaml_parses(self) -> None:
        cfg = load_models_config()
        assert cfg.endpoint.pinned_revision == "local-stub-not-deployed"
        assert cfg.calibration.method == "temperature_scaling"
        assert cfg.calibration.temperature == 1.0

    def test_unknown_key_rejected(self, tmp_path) -> None:
        bad = tmp_path / "thresholds.yaml"
        bad.write_text("version: '9'\nbogus_key: 1\n", encoding="utf-8")
        with pytest.raises(Exception, match="bogus_key"):
            load_thresholds(bad)
