"""Interface en ligne de commande (mode headless).

Usage ::

    python -m panelrecon.cli --input DOSSIER_OU_FICHIER [...] --output DOSSIER [--config cfg.json]
    python -m panelrecon.cli --write-default-config cfg.json

Modes disponibles à ce stade :

* ``reconstruct`` (défaut) : découpe chaque vidéo en séquences (un panel par
  séquence), segmente et reconstruit chaque panel et écrit, dans
  ``<sortie>/<vidéo>/``, un PNG RGBA, une carte de couverture et un rapport JSON
  par séquence, plus ``<vidéo>_sequences.json`` ; ``batch_report.json`` résume
  le lot.

* ``inventory`` : découvre les vidéos, les décode intégralement en flux et écrit
  ``inventory.json`` (métadonnées, nombre de frames, statistiques des
  intervalles de temps pour détecter la fréquence variable, erreurs). Une vidéo
  illisible est journalisée avec sa trace et n'interrompt pas le lot.

* ``register`` : découpe chaque vidéo en séquences, les recale et écrit
  ``<vidéo>_registration.json`` : séquences, transitions (frames écartées),
  transformations frame → repère canonique, estimations acceptées.

Codes de sortie : 0 = succès, 1 = au moins une vidéo en échec,
2 = erreur d'usage ou de configuration, 130 = interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from panelrecon import __version__
from panelrecon.core.config import ConfigError, PipelineConfig
from panelrecon.core.hardware import resolve_num_workers, select_torch_device
from panelrecon.core.models import CancellationToken, OperationCancelled
from panelrecon.core.export import write_json
from panelrecon.core.pipeline import VideoOutcome, process_video
from panelrecon.core.registration import RegistrationResult
from panelrecon.core.scene_split import split_and_register
from panelrecon.core.video_io import VideoError, VideoReader, discover_videos

logger = logging.getLogger("panelrecon.cli")

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

INVENTORY_FILENAME = "inventory.json"
BATCH_REPORT_FILENAME = "batch_report.json"
LOG_FILENAME = "panelrecon.log"
# Écart relatif entre intervalles extrêmes au-delà duquel la vidéo est jugée VFR.
_VFR_RELATIVE_SPREAD = 0.05


def configure_logging(level: str, log_file: Path | None) -> None:
    """Configure la journalisation racine : console + fichier optionnel."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panelrecon",
        description="Reconstruction de panels de manhwa à partir de vidéos (mode headless).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--input", "-i", type=Path, nargs="+", help="Fichiers vidéo et/ou dossiers à traiter."
    )
    parser.add_argument("--output", "-o", type=Path, help="Dossier de sortie.")
    parser.add_argument("--config", "-c", type=Path, help="Fichier de configuration JSON.")
    parser.add_argument(
        "--mode",
        choices=("reconstruct", "inventory", "register"),
        default="reconstruct",
        help="Traitement à exécuter (défaut : reconstruct).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Surcharge runtime.log_level de la configuration.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        metavar="CHEMIN",
        help="Écrit la configuration par défaut en JSON puis quitte.",
    )
    return parser


@dataclass
class VideoInventory:
    path: Path
    status: str
    info: dict[str, Any] | None = None
    frames_decoded: int = 0
    frames_kept: int = 0
    decode_errors: int = 0
    first_time_s: float | None = None
    last_time_s: float | None = None
    interval_min_s: float | None = None
    interval_median_s: float | None = None
    interval_max_s: float | None = None
    variable_frame_rate: bool | None = None
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": self.status,
            "info": self.info,
            "frames_decoded": self.frames_decoded,
            "frames_kept": self.frames_kept,
            "decode_errors": self.decode_errors,
            "first_time_s": self.first_time_s,
            "last_time_s": self.last_time_s,
            "interval_min_s": self.interval_min_s,
            "interval_median_s": self.interval_median_s,
            "interval_max_s": self.interval_max_s,
            "variable_frame_rate": self.variable_frame_rate,
            "elapsed_s": round(self.elapsed_s, 3),
            "error": self.error,
        }


def inventory_video(
    path: Path, config: PipelineConfig, cancel: CancellationToken | None = None
) -> VideoInventory:
    """Décode une vidéo en flux et en résume les propriétés temporelles."""
    start = time.perf_counter()
    entry = VideoInventory(path=path, status="ok")
    try:
        reader = VideoReader(path, config.video, config.preprocess, cancel=cancel)
        entry.info = reader.info.to_dict()
        times: list[float] = []
        with reader:
            for obs in reader.frames(compute_proxy=False):
                times.append(obs.time_s)
        entry.frames_decoded = reader.frames_decoded
        entry.decode_errors = reader.decode_errors
        entry.frames_kept = len(times)
        if times:
            entry.first_time_s = times[0]
            entry.last_time_s = times[-1]
        if len(times) >= 2:
            intervals = np.diff(np.asarray(times, dtype=np.float64))
            entry.interval_min_s = float(intervals.min())
            entry.interval_median_s = float(np.median(intervals))
            entry.interval_max_s = float(intervals.max())
            if entry.interval_median_s > 0:
                spread = (entry.interval_max_s - entry.interval_min_s) / entry.interval_median_s
                entry.variable_frame_rate = bool(spread > _VFR_RELATIVE_SPREAD)
        if entry.frames_kept == 0:
            entry.status = "error"
            entry.error = "Aucune frame décodable"
            logger.error("%s : aucune frame décodable", path)
    except OperationCancelled:
        raise
    except (VideoError, OSError, ValueError) as exc:
        entry.status = "error"
        entry.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Échec de lecture de %s", path)
    entry.elapsed_s = time.perf_counter() - start
    return entry


def run_inventory(
    inputs: Sequence[Path],
    output_dir: Path,
    config: PipelineConfig,
    cancel: CancellationToken | None = None,
) -> list[VideoInventory]:
    videos = discover_videos(inputs, config.video.extensions, config.video.recursive)
    logger.info("%d vidéo(s) trouvée(s)", len(videos))
    results: list[VideoInventory] = []
    for n, path in enumerate(videos, start=1):
        logger.info("[%d/%d] %s", n, len(videos), path)
        entry = inventory_video(path, config, cancel)
        if entry.status == "ok":
            logger.info(
                "  %d frames, %.2f s, VFR=%s, %d erreur(s) de décodage",
                entry.frames_kept,
                (entry.last_time_s or 0.0) - (entry.first_time_s or 0.0),
                entry.variable_frame_rate,
                entry.decode_errors,
            )
        results.append(entry)
    report = {
        "panelrecon_version": __version__,
        "config": config.to_dict(),
        "videos": [r.to_dict() for r in results],
        "n_ok": sum(r.status == "ok" for r in results),
        "n_failed": sum(r.status != "ok" for r in results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / INVENTORY_FILENAME
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    logger.info("Inventaire écrit : %s", out_path)
    return results


def registration_report(result: RegistrationResult) -> dict[str, Any]:
    return {
        "reference_index": result.reference_index,
        "canonical_scale": result.canonical_scale,
        "mean_inlier_ratio": result.mean_inlier_ratio,
        "min_inlier_ratio": result.min_inlier_ratio,
        "rms_reprojection_error": result.rms_reprojection_error,
        "excluded": result.excluded,
        "transforms": {str(i): t.to_dict() for i, t in sorted(result.transforms.items())},
        "estimates": [e.to_dict() for e in result.estimates],
        "rejected": [e.to_dict() for e in result.rejected],
    }


def run_register(
    inputs: Sequence[Path],
    output_dir: Path,
    config: PipelineConfig,
    cancel: CancellationToken | None = None,
) -> list[VideoInventory]:
    """Découpe et recale chaque vidéo ; une erreur n'arrête pas le lot."""
    videos = discover_videos(inputs, config.video.extensions, config.video.recursive)
    logger.info("%d vidéo(s) trouvée(s)", len(videos))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[VideoInventory] = []
    for n, path in enumerate(videos, start=1):
        logger.info("[%d/%d] %s", n, len(videos), path)
        start = time.perf_counter()
        entry = VideoInventory(path=path, status="ok")
        try:
            with VideoReader(path, config.video, config.preprocess, cancel=cancel) as reader:
                entry.info = reader.info.to_dict()
                split = split_and_register(reader.frames(), config, cancel)
                entry.frames_decoded = reader.frames_decoded
                entry.decode_errors = reader.decode_errors
            entry.frames_kept = sum(len(s.registration.transforms) for s in split.sequences)
            report = {
                "panelrecon_version": __version__,
                "video": entry.info,
                "sequences": [
                    {"sequence": s.sequence.to_dict(), **registration_report(s.registration)}
                    for s in split.sequences
                ],
                "transitions": [t.to_dict() for t in split.transitions],
                "dropped_sequences": [s.to_dict() for s in split.dropped],
                "discarded_frames": split.discarded,
            }
            write_json(output_dir / f"{path.stem}_registration.json", report)
            if not split.sequences:
                entry.status = "error"
                entry.error = "Aucune séquence exploitable"
        except OperationCancelled:
            raise
        except (VideoError, OSError, ValueError) as exc:
            entry.status = "error"
            entry.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Échec du recalage de %s", path)
        entry.elapsed_s = time.perf_counter() - start
        results.append(entry)
    return results


def run_reconstruct(
    inputs: Sequence[Path],
    output_dir: Path,
    config: PipelineConfig,
    cancel: CancellationToken | None = None,
) -> list[VideoOutcome]:
    """Reconstruit chaque vidéo ; une vidéo en échec est rapportée sans arrêter le lot."""
    videos = discover_videos(inputs, config.video.extensions, config.video.recursive)
    logger.info("%d vidéo(s) trouvée(s)", len(videos))
    outcomes: list[VideoOutcome] = []
    for n, path in enumerate(videos, start=1):
        logger.info("[%d/%d] %s", n, len(videos), path)
        outcome = process_video(path, output_dir, config, cancel=cancel)
        logger.info("  %s en %.1f s", "OK" if outcome.ok else f"ÉCHEC : {outcome.error}",
                    outcome.elapsed_s)
        outcomes.append(outcome)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / BATCH_REPORT_FILENAME, {
        "panelrecon_version": __version__,
        "config": config.to_dict(),
        "videos": [o.to_dict() for o in outcomes],
        "n_ok": sum(o.ok for o in outcomes),
        "n_failed": sum(not o.ok for o in outcomes),
    })
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_default_config is not None:
        configure_logging(args.log_level or "INFO", None)
        PipelineConfig().save(args.write_default_config)
        logger.info("Configuration par défaut écrite : %s", args.write_default_config)
        return EXIT_OK

    if not args.input or args.output is None:
        configure_logging(args.log_level or "INFO", None)
        parser.print_usage(sys.stderr)
        logger.error("--input et --output sont requis")
        return EXIT_USAGE

    try:
        config = PipelineConfig.load(args.config) if args.config else PipelineConfig()
        config.validate()
    except ConfigError as exc:
        configure_logging(args.log_level or "INFO", None)
        logger.error("Configuration invalide : %s", exc)
        return EXIT_USAGE

    output_dir: Path = args.output.expanduser()
    configure_logging(args.log_level or config.runtime.log_level, output_dir / LOG_FILENAME)
    device = select_torch_device(config.runtime.device)
    logger.info(
        "panelrecon %s — device modules optionnels : %s (%s), workers : %d",
        __version__,
        device.name,
        device.reason,
        resolve_num_workers(config.runtime.num_workers),
    )

    cancel = CancellationToken()
    try:
        if args.mode == "reconstruct":
            outcomes = run_reconstruct(args.input, output_dir, config, cancel)
            statuses = [o.ok for o in outcomes]
        else:
            runner = run_register if args.mode == "register" else run_inventory
            statuses = [r.status == "ok" for r in runner(args.input, output_dir, config, cancel)]
    except (KeyboardInterrupt, OperationCancelled):
        logger.warning("Traitement interrompu")
        return EXIT_INTERRUPTED
    if not statuses:
        logger.error("Aucune vidéo trouvée dans les entrées fournies")
        return EXIT_FAILURES
    return EXIT_OK if all(statuses) else EXIT_FAILURES


if __name__ == "__main__":
    sys.exit(main())
