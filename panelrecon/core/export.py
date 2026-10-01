"""Export : PNG RGBA du panel, carte de couverture, rapport JSON.

Arborescence produite pour une vidéo ``<nom>`` ::

    <sortie>/<nom>/<nom>_seq000_<début>-<fin>.png            panel reconstruit (RGBA)
    <sortie>/<nom>/<nom>_seq000_<début>-<fin>_coverage.png   couverture brute (16 bits)
    <sortie>/<nom>/<nom>_seq000_<début>-<fin>_coverage_color.png  couverture en fausses couleurs
    <sortie>/<nom>/<nom>_seq000_<début>-<fin>.json           rapport

Toutes les écritures sont atomiques (fichier temporaire puis renommage).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from panelrecon import __version__
from panelrecon.core.config import PipelineConfig
from panelrecon.core.models import ImageU8, MosaicResult, Sequence, VideoInfo
from panelrecon.core.registration import RegistrationResult

logger = logging.getLogger(__name__)

REPORT_VERSION = 1


@dataclass(frozen=True)
class ExportedFiles:
    image: Path
    report: Path
    coverage: Path | None = None
    coverage_color: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "image": str(self.image),
            "report": str(self.report),
            "coverage": None if self.coverage is None else str(self.coverage),
            "coverage_color": None if self.coverage_color is None else str(self.coverage_color),
        }


def sequence_basename(video_stem: str, number: int, sequence: Sequence) -> str:
    return f"{video_stem}_seq{number:03d}_{sequence.start_idx:06d}-{sequence.end_idx:06d}"


def write_png(path: Path, image: NDArray[Any], compression: int = 3) -> Path:
    """Écrit un PNG (BGR, BGRA ou niveaux de gris, 8 ou 16 bits) de façon atomique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    ok = cv2.imwrite(str(tmp), image, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Écriture PNG impossible : {path}")
    tmp.replace(path)
    return path


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def coverage_colormap(coverage: NDArray[np.uint16]) -> ImageU8:
    """Couverture en fausses couleurs (TURBO) ; pixels jamais observés en noir."""
    peak = max(1, int(coverage.max()))
    normalized = np.clip(np.rint(coverage.astype(np.float32) * (255.0 / peak)), 0, 255)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[coverage == 0] = 0
    return np.asarray(colored, dtype=np.uint8)


def coverage_statistics(result: MosaicResult) -> dict[str, float | int]:
    covered = result.image_bgra[..., 3] > 0
    values = result.coverage[covered]
    return {
        "covered_pixels": int(covered.sum()),
        "crop_pixels": int(covered.size),
        "coverage_ratio": float(covered.mean()),
        "observations_min": int(values.min()) if values.size else 0,
        "observations_mean": float(values.mean()) if values.size else 0.0,
        "observations_max": int(values.max()) if values.size else 0,
    }


def mosaic_report(
    result: MosaicResult,
    registration: RegistrationResult,
    config: PipelineConfig,
    video: VideoInfo | None = None,
    files: ExportedFiles | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "panelrecon_version": __version__,
        "video": None if video is None else video.to_dict(),
        "sequence": result.sequence.to_dict(),
        "image": {"width": result.crop.width, "height": result.crop.height},
        "canvas_scale": result.canvas_scale,
        "crop": result.crop.to_dict(),
        "frames_used": len(result.transforms),
        "coverage": coverage_statistics(result),
        "registration": {
            "reference_index": registration.reference_index,
            "interrupted": registration.interrupted,
            "interruption_reason": registration.interruption_reason,
            "mean_inlier_ratio": registration.mean_inlier_ratio,
            "min_inlier_ratio": registration.min_inlier_ratio,
            "rms_reprojection_error": registration.rms_reprojection_error,
            "excluded": registration.excluded,
            "rejected": [e.to_dict() for e in registration.rejected],
        },
    }
    if config.export.save_registration:
        report["registration"]["transforms"] = {
            str(i): t.to_dict() for i, t in sorted(result.transforms.items())
        }
        report["registration"]["estimates"] = [e.to_dict() for e in registration.estimates]
    if files is not None:
        report["files"] = files.to_dict()
    if extra:
        report.update(extra)
    return report


def export_sequence(
    result: MosaicResult,
    registration: RegistrationResult,
    out_dir: Path,
    basename: str,
    config: PipelineConfig,
    video: VideoInfo | None = None,
    extra: dict[str, Any] | None = None,
) -> ExportedFiles:
    """Écrit le panel, la couverture et le rapport d'une séquence."""
    compression = config.export.png_compression
    image = write_png(out_dir / f"{basename}.png", result.image_bgra, compression)
    coverage = coverage_color = None
    if config.export.save_coverage_map:
        coverage = write_png(out_dir / f"{basename}_coverage.png", result.coverage, compression)
        coverage_color = write_png(out_dir / f"{basename}_coverage_color.png",
                                   coverage_colormap(result.coverage), compression)
    files = ExportedFiles(image, out_dir / f"{basename}.json", coverage, coverage_color)
    write_json(files.report, mosaic_report(result, registration, config, video, files, extra))
    logger.info("Exporté : %s", image)
    return files
