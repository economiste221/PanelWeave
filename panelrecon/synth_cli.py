"""Génération de vidéos synthétiques de référence (avec vérité terrain).

Usage ::

    python -m panelrecon.synth_cli --list
    python -m panelrecon.synth_cli --output dossier --all [--seed 0]
    python -m panelrecon.synth_cli --output dossier --scenario zoom_in crossfade
    python -m panelrecon.synth_cli --output dossier --scenario pan_zoom_eased --panel-image mon_panel.png

Chaque scénario produit ``<nom>.<ext>`` (vidéo), ``<nom>_panel<k>.png`` (panels
originaux) et ``<nom>_ground_truth.json``. Avec ``--oracle``, chaque plan est en
plus reconstruit avec les poses exactes et ses métriques sont affichées : c'est
la borne supérieure attendue du pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from panelrecon.cli import configure_logging
from panelrecon.core.config import PipelineConfig
from panelrecon.core.evaluation import ReferenceThresholds, evaluate_mosaic, oracle_mosaic
from panelrecon.core.synthetic import SCENARIOS, VideoSpec, generate_video, scenario
from panelrecon.core.video_io import VideoReader

logger = logging.getLogger("panelrecon.synth_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panelrecon-synth", description="Vidéos synthétiques avec vérité terrain."
    )
    parser.add_argument("--output", "-o", type=Path, help="Dossier de sortie.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", "-s", nargs="+", choices=sorted(SCENARIOS))
    group.add_argument("--all", action="store_true", help="Tous les scénarios.")
    parser.add_argument("--list", action="store_true", help="Lister les scénarios et quitter.")
    parser.add_argument("--seed", type=int, default=0, help="Graine des panels procéduraux.")
    parser.add_argument(
        "--panel-image", type=Path, help="Image à utiliser comme panel (au lieu du procédural)."
    )
    parser.add_argument(
        "--oracle", action="store_true", help="Reconstruire avec les poses exactes et évaluer."
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def _with_panel_image(spec: VideoSpec, image: Path) -> VideoSpec:
    shots = tuple(replace(s, panel=replace(s.panel, image_path=image)) for s in spec.shots)
    return replace(spec, shots=shots)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, None)
    if args.list:
        for name in sorted(SCENARIOS):
            spec = scenario(name, args.seed)
            n = sum(s.n_frames for s in spec.shots)
            logger.info("%-16s %d plan(s), %d frames de plan", name, len(spec.shots), n)
        return 0
    if args.output is None or not (args.all or args.scenario):
        parser.print_usage(sys.stderr)
        logger.error("--output et (--scenario | --all) sont requis")
        return 2
    if args.panel_image is not None and not args.panel_image.is_file():
        logger.error("Image de panel introuvable : %s", args.panel_image)
        return 2
    names = sorted(SCENARIOS) if args.all else args.scenario
    config = PipelineConfig()
    thresholds = ReferenceThresholds()
    failures = 0
    for name in names:
        spec = scenario(name, args.seed)
        if args.panel_image is not None:
            spec = _with_panel_image(spec, args.panel_image.resolve())
        gt = generate_video(spec, args.output)
        if not args.oracle:
            continue
        for shot_id in range(len(gt.shots)):
            with VideoReader(gt.video_path, config.video, config.preprocess) as reader:
                frames = ((f.index, f.image) for f in reader.frames(compute_proxy=False))
                result = oracle_mosaic(frames, gt, shot_id)
            metrics = evaluate_mosaic(result, gt, shot_id)
            problems = metrics.check(thresholds)
            failures += bool(problems)
            logger.info(
                "%s plan %d : SSIM %.4f, PSNR %.1f dB, couverture %.4f, surestimation %.4f %s",
                name, shot_id, metrics.ssim, metrics.psnr_db, metrics.true_coverage,
                metrics.coverage_overreach, "OK" if not problems else f"ÉCHEC {problems}",
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
