"""Modèles de données du pipeline.

Convention de coordonnées (valable dans tout le paquet) :

* coordonnées pixel ``(x, y)``, origine au **centre du pixel en haut à gauche**,
  x vers la droite, y vers le bas — c'est la convention de ``cv2.warpAffine`` ;
* une :class:`SimilarityTransform` associée à une frame transforme les
  coordonnées **de la frame** vers les coordonnées **du repère cible**
  (frame suivante, mosaïque ou canevas canonique selon le contexte).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageU8: TypeAlias = NDArray[np.uint8]
MaskU8: TypeAlias = NDArray[np.uint8]
FloatArray: TypeAlias = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Annulation coopérative
# ---------------------------------------------------------------------------


class OperationCancelled(Exception):
    """Levée lorsqu'un traitement est interrompu à la demande de l'utilisateur."""


class CancelFlag(Protocol):
    """Tout objet exposant ``is_set()`` (threading.Event, multiprocessing.Event…)."""

    def is_set(self) -> bool: ...


class CancellationToken:
    """Drapeau d'annulation vérifié entre les frames.

    Peut envelopper un ``multiprocessing.Event`` pour une annulation inter-processus.
    """

    def __init__(self, flag: CancelFlag | None = None) -> None:
        self._flag: CancelFlag = flag if flag is not None else threading.Event()

    def cancel(self) -> None:
        setter = getattr(self._flag, "set", None)
        if setter is None:
            raise TypeError("Le drapeau sous-jacent ne supporte pas set()")
        setter()

    @property
    def cancelled(self) -> bool:
        return bool(self._flag.is_set())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled("Traitement annulé")


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


class NotASimilarityError(ValueError):
    """La matrice fournie s'écarte trop d'une similarité 2D."""


@dataclass(frozen=True)
class SimilarityTransform:
    """Similarité 2D : ``p' = s·R(θ)·p + t``.

    Matrice 2x3 : ``[[s·cosθ, −s·sinθ, tx], [s·sinθ, s·cosθ, ty]]``.
    """

    scale: float = 1.0
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        values = (self.scale, self.theta, self.tx, self.ty)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Paramètres de similarité non finis : {values}")
        if self.scale <= 0.0:
            raise ValueError(f"Échelle non strictement positive : {self.scale}")

    # ------------------------------------------------------------ constructeurs
    @classmethod
    def identity(cls) -> SimilarityTransform:
        return cls()

    @classmethod
    def from_matrix(cls, matrix: NDArray[Any], tol: float = 1e-3) -> SimilarityTransform:
        """Construit la similarité la plus proche d'une matrice 2x3 (ou 3x3 affine).

        Lève :class:`NotASimilarityError` si la partie linéaire s'écarte d'une
        similarité de plus de ``tol`` (relativement à l'échelle).
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (3, 3):
            if not np.allclose(m[2], (0.0, 0.0, 1.0), atol=1e-9):
                raise NotASimilarityError("Matrice 3x3 projective, pas une similarité")
            m = m[:2]
        if m.shape != (2, 3):
            raise ValueError(f"Forme de matrice invalide : {m.shape}")
        if not np.all(np.isfinite(m)):
            raise NotASimilarityError("Matrice non finie")
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        # Projection orthogonale sur {[[p, -q], [q, p]]}.
        p = 0.5 * (a + d)
        q = 0.5 * (c - b)
        scale = math.hypot(p, q)
        if scale <= 1e-12:
            raise NotASimilarityError("Échelle nulle")
        residual = math.hypot(a - d, b + c) / (2.0 * scale)
        if residual > tol:
            raise NotASimilarityError(
                f"Écart à une similarité trop grand ({residual:.2e} > {tol:.2e})"
            )
        return cls(scale=scale, theta=math.atan2(q, p), tx=float(m[0, 2]), ty=float(m[1, 2]))

    @classmethod
    def fit(
        cls,
        src: NDArray[Any],
        dst: NDArray[Any],
        weights: NDArray[Any] | None = None,
        allow_rotation: bool = True,
        rotation_regularization: float = 0.0,
    ) -> SimilarityTransform:
        """Similarité minimisant ``Σ wᵢ‖T(srcᵢ) − dstᵢ‖² + λ·σ²·b²`` (forme fermée d'Umeyama).

        ``T`` a pour partie linéaire ``[[a, −b], [b, a]]`` ; ``σ²`` est la variance
        pondérée des sources. ``λ = rotation_regularization`` pénalise la rotation
        (``b`` est divisé par ``1 + λ``) ; ``allow_rotation=False`` la fixe à 0.
        """
        if rotation_regularization < 0:
            raise ValueError("rotation_regularization doit être ≥ 0")
        p = np.asarray(src, dtype=np.float64)
        q = np.asarray(dst, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 2 or p.shape != q.shape:
            raise ValueError(f"Points (N, 2) appariés attendus, reçu {p.shape} et {q.shape}")
        if p.shape[0] < 2:
            raise ValueError("Au moins deux correspondances sont nécessaires")
        w = np.ones(p.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (p.shape[0],) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Poids invalides")
        w = w / w.sum()
        mu_p = w @ p
        mu_q = w @ q
        pc = p - mu_p
        qc = q - mu_q
        var_p = float(w @ (pc**2).sum(axis=1))
        if var_p <= 1e-18:
            raise ValueError("Points sources dégénérés (tous confondus)")
        # Composantes de la covariance croisée utiles pour [[a, -b], [b, a]].
        sxx = float(w @ (pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1]))
        sxy = float(w @ (pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]))
        if allow_rotation:
            a, b = sxx / var_p, sxy / (var_p * (1.0 + rotation_regularization))
        else:
            a, b = sxx / var_p, 0.0
        scale = math.hypot(a, b)
        if scale <= 1e-12:
            raise ValueError("Échelle estimée nulle")
        theta = math.atan2(b, a)
        tx = float(mu_q[0] - (a * mu_p[0] - b * mu_p[1]))
        ty = float(mu_q[1] - (b * mu_p[0] + a * mu_p[1]))
        return cls(scale=scale, theta=theta, tx=tx, ty=ty)

    @classmethod
    def from_translation(cls, tx: float, ty: float) -> SimilarityTransform:
        return cls(1.0, 0.0, tx, ty)

    @classmethod
    def from_scale_about(cls, scale: float, cx: float, cy: float) -> SimilarityTransform:
        """Zoom de facteur ``scale`` autour du point ``(cx, cy)``."""
        return cls(scale, 0.0, cx - scale * cx, cy - scale * cy)

    # ------------------------------------------------------------------ matrices
    def matrix(self) -> FloatArray:
        c = self.scale * math.cos(self.theta)
        s = self.scale * math.sin(self.theta)
        return np.array([[c, -s, self.tx], [s, c, self.ty]], dtype=np.float64)

    def homogeneous(self) -> FloatArray:
        h = np.eye(3, dtype=np.float64)
        h[:2] = self.matrix()
        return h

    @property
    def log_scale(self) -> float:
        return math.log(self.scale)

    # --------------------------------------------------------------- opérations
    def __matmul__(self, other: SimilarityTransform) -> SimilarityTransform:
        """Composition : ``(A @ B)(p) = A(B(p))``."""
        if not isinstance(other, SimilarityTransform):
            return NotImplemented
        return SimilarityTransform.from_matrix(self.homogeneous() @ other.homogeneous(), tol=1e-6)

    def inverse(self) -> SimilarityTransform:
        inv_scale = 1.0 / self.scale
        c = inv_scale * math.cos(-self.theta)
        s = inv_scale * math.sin(-self.theta)
        tx = -(c * self.tx - s * self.ty)
        ty = -(s * self.tx + c * self.ty)
        return SimilarityTransform(inv_scale, -self.theta, tx, ty)

    def apply(self, points: NDArray[Any]) -> FloatArray:
        """Applique la transformation à un tableau de points ``(N, 2)``."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Points de forme (N, 2) attendus, reçu {pts.shape}")
        m = self.matrix()
        out: FloatArray = pts @ m[:, :2].T + m[:, 2]
        return out

    def rescaled(self, src_factor: float, dst_factor: float | None = None) -> SimilarityTransform:
        """Convertit une transformation estimée sur des images réduites vers la résolution native.

        Si la transformation ``M`` a été estimée entre une source réduite de
        ``src_factor`` (``p_proxy = src_factor · p_natif``) et une cible réduite
        de ``dst_factor`` (par défaut ``src_factor``), alors la transformation
        native est la conjugaison ``S_dst⁻¹ · M · S_src``.
        """
        if dst_factor is None:
            dst_factor = src_factor
        if src_factor <= 0.0 or dst_factor <= 0.0:
            raise ValueError("Les facteurs d'échelle doivent être strictement positifs")
        return SimilarityTransform(
            scale=self.scale * src_factor / dst_factor,
            theta=self.theta,
            tx=self.tx / dst_factor,
            ty=self.ty / dst_factor,
        )

    def to_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "theta": self.theta, "tx": self.tx, "ty": self.ty}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimilarityTransform:
        return cls(
            scale=float(data["scale"]),
            theta=float(data["theta"]),
            tx=float(data["tx"]),
            ty=float(data["ty"]),
        )


class MotionMethod(str, Enum):
    """Méthode ayant produit une estimation de mouvement."""

    SIFT = "sift"
    ORB = "orb"
    LOFTR = "loftr"
    RAFT = "raft"
    FARNEBACK = "farneback"
    PHASE_CORRELATION = "phase_correlation"
    MOSAIC = "mosaic"
    GLOBAL_ADJUSTMENT = "global_adjustment"


@dataclass(frozen=True)
class MotionEstimate:
    """Résultat d'une estimation de mouvement entre deux images.

    ``transform`` envoie les coordonnées de la source vers celles de la cible.
    """

    src_index: int
    dst_index: int
    transform: SimilarityTransform
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    rms_reprojection_error: float
    method: MotionMethod
    ecc_refined: bool = False
    accepted: bool = True
    reason: str = ""
    ncc: float | None = None  # corrélation photométrique sur le recouvrement

    def __post_init__(self) -> None:
        if self.n_matches < 0 or self.n_inliers < 0 or self.n_inliers > self.n_matches:
            raise ValueError(
                f"Comptes incohérents : {self.n_inliers} inliers / {self.n_matches} appariements"
            )
        if not 0.0 <= self.inlier_ratio <= 1.0:
            raise ValueError(f"inlier_ratio hors de [0, 1] : {self.inlier_ratio}")
        if not (self.rms_reprojection_error >= 0.0):
            raise ValueError(f"Erreur RMS invalide : {self.rms_reprojection_error}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_index": self.src_index,
            "dst_index": self.dst_index,
            "transform": self.transform.to_dict(),
            "n_matches": self.n_matches,
            "n_inliers": self.n_inliers,
            "inlier_ratio": self.inlier_ratio,
            "rms_reprojection_error": self.rms_reprojection_error,
            "method": self.method.value,
            "ecc_refined": self.ecc_refined,
            "accepted": self.accepted,
            "reason": self.reason,
            "ncc": self.ncc,
        }


# ---------------------------------------------------------------------------
# Frames et séquences
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class FrameObs:
    """Une frame décodée.

    ``image`` est en BGR uint8, à la résolution native sauf en lecture « image
    réduite seule », où elle est à la taille de ``proxy_gray`` et où
    ``native_size`` donne la taille native. ``proxy_gray`` est la version réduite
    en niveaux de gris utilisée pour l'estimation du mouvement, avec
    ``p_proxy = proxy_factor · p_natif``. Les transformations sont toujours
    exprimées en coordonnées **natives**.
    """

    index: int
    time_s: float
    pts: int | None
    image: ImageU8
    proxy_gray: ImageU8 | None = None
    proxy_factor: float = 1.0
    native_size: tuple[int, int] | None = None  # (largeur, hauteur) si image réduite

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Index de frame négatif : {self.index}")
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(
                f"Image BGR uint8 (H, W, 3) attendue, reçu {self.image.dtype} {self.image.shape}"
            )
        if not 0.0 < self.proxy_factor <= 1.0:
            raise ValueError(f"proxy_factor hors de ]0, 1] : {self.proxy_factor}")
        if self.proxy_gray is not None and self.proxy_gray.ndim != 2:
            raise ValueError("proxy_gray doit être une image 2D")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def native_width(self) -> int:
        return self.native_size[0] if self.native_size is not None else self.width

    @property
    def native_height(self) -> int:
        return self.native_size[1] if self.native_size is not None else self.height

    @property
    def is_native(self) -> bool:
        return self.native_size is None or self.native_size == (self.width, self.height)


@dataclass(frozen=True)
class Sequence:
    """Plage de frames correspondant à un même panel (bornes **incluses**).

    ``excluded`` liste les indices écartés de la fusion (fondus, frames aberrantes).
    """

    start_idx: int
    end_idx: int
    excluded: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError(f"Bornes de séquence invalides : [{self.start_idx}, {self.end_idx}]")
        for idx in self.excluded:
            if not self.start_idx <= idx <= self.end_idx:
                raise ValueError(f"Indice exclu {idx} hors de la séquence")

    def __len__(self) -> int:
        return self.end_idx - self.start_idx + 1

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and self.start_idx <= index <= self.end_idx

    @property
    def usable_indices(self) -> tuple[int, ...]:
        excluded = set(self.excluded)
        return tuple(i for i in range(self.start_idx, self.end_idx + 1) if i not in excluded)

    def to_dict(self) -> dict[str, Any]:
        return {"start_idx": self.start_idx, "end_idx": self.end_idx, "excluded": list(self.excluded)}


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropBox:
    """Rectangle ``[x0, x1) × [y0, y1)`` en pixels du canevas."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"Rectangle vide : {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(eq=False)
class MosaicResult:
    """Panel reconstruit pour une séquence.

    ``transforms[i]`` envoie les coordonnées natives de la frame ``i`` vers les
    coordonnées du canevas **avant** recadrage. ``image_bgra`` (canaux dans l'ordre
    OpenCV ; alpha = 0 sur les pixels jamais observés) et ``coverage`` (nombre
    d'observations valides par pixel) sont déjà recadrés sur ``crop``.
    """

    sequence: Sequence
    image_bgra: ImageU8
    coverage: NDArray[np.uint16]
    transforms: dict[int, SimilarityTransform]
    crop: CropBox
    canvas_scale: float

    def __post_init__(self) -> None:
        h, w = self.coverage.shape
        if self.image_bgra.shape != (h, w, 4) or self.image_bgra.dtype != np.uint8:
            raise ValueError("image_bgra doit être (H, W, 4) uint8, de même taille que coverage")
        if (self.crop.height, self.crop.width) != (h, w):
            raise ValueError("Les dimensions de crop ne correspondent pas à l'image")
        if self.canvas_scale <= 0.0:
            raise ValueError("canvas_scale doit être strictement positif")


class Verdict(str, Enum):
    OK = "OK"
    TO_REVIEW = "À VÉRIFIER"
    FAILED = "ÉCHEC"


@dataclass(frozen=True)
class QualityReport:
    """Métriques de qualité d'une séquence reconstruite."""

    sequence: Sequence
    n_frames_used: int
    mean_inlier_ratio: float
    min_inlier_ratio: float
    rms_reprojection_error: float
    coverage_ratio: float
    mean_ssim: float
    sharpness: float
    verdict: Verdict
    reasons: tuple[str, ...] = field(default_factory=tuple)
    min_ssim: float | None = None  # pire frame gardée
    frames_evaluated: int = 0
    excluded_frames: tuple[int, ...] = ()  # écartées par le contrôle qualité
    frame_ssim: tuple[tuple[int, float], ...] = ()  # (frame, SSIM) de chaque frame évaluée

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence.to_dict(),
            "n_frames_used": self.n_frames_used,
            "mean_inlier_ratio": self.mean_inlier_ratio,
            "min_inlier_ratio": self.min_inlier_ratio,
            "rms_reprojection_error": self.rms_reprojection_error,
            "coverage_ratio": self.coverage_ratio,
            "mean_ssim": self.mean_ssim,
            "sharpness": self.sharpness,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "min_ssim": self.min_ssim,
            "frames_evaluated": self.frames_evaluated,
            "excluded_frames": list(self.excluded_frames),
            "frame_ssim": {str(i): round(v, 4) for i, v in self.frame_ssim},
        }


@dataclass(frozen=True)
class VideoInfo:
    """Métadonnées d'une vidéo obtenues sans décodage complet."""

    path: Path
    width: int
    height: int
    fps: float
    duration_s: float | None
    frame_count: int | None
    codec: str
    backend: str
    rotation_deg: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "codec": self.codec,
            "backend": self.backend,
            "rotation_deg": self.rotation_deg,
        }
