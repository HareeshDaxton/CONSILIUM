"""Router tests (§4.2): grant, deny-by-default, glob, hash stability."""

import hashlib
from pathlib import Path

from consilium.context.builder import build
from consilium.context.router import load_routing_config, route
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import FactSource, Laterality, QualityFlag, Role, ScreeningTier
from consilium.schemas.vision import (
    CADResult,
    CDRMetrics,
    ClassifierOutput,
    ImageQuality,
    MaskQC,
    Measurements,
)

ROUTING_YAML = Path("configs/routing.yaml")


def make_ctx() -> GlobalContext:
    cad = CADResult(
        quality=ImageQuality(score=0.9, gradable=True, flags=(QualityFlag.BLURRED,)),
        classifier=ClassifierOutput(
            p_raw=0.7, p_calibrated=0.62, tier=ScreeningTier.HIGH, model_version="m@r"
        ),
        measurements=Measurements(
            disc_area_px=10000,
            cup_area_px=2500,
            disc_vertical_diameter_px=120,
            cup_vertical_diameter_px=60,
        ),
        cdr=CDRMetrics(vertical=0.5, area_based=0.5),
        mask_qc=MaskQC(
            cup_within_disc=True,
            single_component_each=True,
            disc_area_plausible=True,
            ok=True,
        ),
        segmenter_version="s@r",
        laterality=Laterality.OD,
    )
    return build(cad, None, None, "case-1")


def test_load_real_config_and_hash_matches_file_bytes() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    assert cfg.version == hashlib.sha256(ROUTING_YAML.read_bytes()).hexdigest()
    assert set(cfg.roles) == {Role.OPHTHALMOLOGIST, Role.OPTOMETRIST, Role.PHARMACIST}


def test_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text("roles:\n  pharmacist:\n    allow: [cad.laterality]\n", encoding="utf-8")
    v1 = load_routing_config(p).version
    p.write_text("roles:\n  pharmacist:\n    allow: [cad.laterality, note.age]\n", encoding="utf-8")
    v2 = load_routing_config(p).version
    assert v1 != v2


def test_grant_and_glob() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    routed = route(make_ctx(), cfg)
    oph = routed[Role.OPHTHALMOLOGIST]
    assert "cad.cdr.vertical" in oph.allowed_ids()
    assert "cad.caveat.blurred" in oph.allowed_ids()  # glob cad.caveat.*
    assert oph.routing_version == cfg.version


def test_deny_by_default_unknown_fact_reaches_nobody() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    ctx = make_ctx()
    stray = Fact(id="cad.experimental.new_metric", label="x", value=1.0, source=FactSource.CAD)
    ctx = GlobalContext(case_id=ctx.case_id, laterality=ctx.laterality, facts=(*ctx.facts, stray))
    routed = route(ctx, cfg)
    for rc in routed.values():
        assert "cad.experimental.new_metric" not in rc.allowed_ids()


def test_pharmacist_isolation_per_policy_table() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    pharm = route(make_ctx(), cfg)[Role.PHARMACIST]
    denied = {
        "cad.p_glaucoma.calibrated",
        "cad.cdr.vertical",
        "cad.cdr.area_based",
        "cad.disc.area_px",
        "cad.cup.area_px",
        "cad.disc.vdiam_px",
        "cad.cup.vdiam_px",
        "cad.quality.score",
        "cad.quality.flags",
        "cad.maskqc.ok",
        "note.free_text",
        "note.exam_findings",
        "cad.caveat.blurred",
    }
    assert denied.isdisjoint(pharm.allowed_ids())
    assert "cad.p_glaucoma.tier" in pharm.allowed_ids()
    assert "cad.laterality" in pharm.allowed_ids()


def test_optometrist_gets_no_pixel_measurements() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    opto = route(make_ctx(), cfg)[Role.OPTOMETRIST]
    assert "cad.cdr.vertical" in opto.allowed_ids()
    assert "cad.disc.area_px" not in opto.allowed_ids()
    assert "cad.cup.vdiam_px" not in opto.allowed_ids()


def test_fact_order_preserved() -> None:
    cfg = load_routing_config(ROUTING_YAML)
    ctx = make_ctx()
    routed = route(ctx, cfg)
    global_order = [f.id for f in ctx.facts]
    for rc in routed.values():
        role_ids = [f.id for f in rc.facts]
        assert role_ids == [i for i in global_order if i in set(role_ids)]
