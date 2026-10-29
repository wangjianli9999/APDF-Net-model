# -*- coding: utf-8 -*-
"""Single-image or folder LNC inference with saved Attention-TDAE and APDF-Net.

Run: python test.py
     python test.py --image /path/to/image.tif --model-dir /path/to/models
     python test.py --image /path/to/folder

Input contract: a ready-to-use four-band TIFF, in order 555, 660, 720, 840 nm.
The supplied image is already registered, background-removed and cropped.
Zero is background. Processing starts with feature extraction from this TIFF;
no image registration, calibration, filtering or cropping is performed here.
No target labels are used.
Band wavelengths cannot be inferred from pixel values; the order is a required
property of the input, inherited from the preprocessing pipeline.

Feature definitions and CSV precision match the training feature tables.
Complete network definitions are included below. All scalers are only
transformed, never fitted. APDF-Net was trained on unscaled N content (%),
so no output inverse transformation or multiplication by 100 is applied.

Dependencies and reproducibility instructions: README.txt.
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from skimage.feature import graycomatrix
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "test data"
BANDS = ("555nm", "660nm", "720nm", "840nm")
TEXTURES = ("MEA", "VAR", "HOM", "CON", "DIS", "ENT", "ASM", "Cor")
VEGETATION_INDICES = (
    "NDVI", "GNDVI", "NDRE", "RVI1", "RVI2", "RVI3", "RVI4", "RVI5",
    "RVI6", "GRVI", "DATT", "MNDI", "MTCI", "RERVI", "RESR",
)
FEATURE_NAMES = [f"{name}_{band}" for band in BANDS for name in TEXTURES]
FEATURE_NAMES += list(VEGETATION_INDICES)
LATENT_NAMES = [f"Latent_Feature_{i}" for i in range(1, 6)]
MODEL_FILES = (
    "attention_tdae_weights_5D.pth", "scaler_X.pkl",
    "apdf_net_best.pth", "apdf_net_scaler.pkl", "apdf_net_meta.json",
)


class FeatureAttention(nn.Module):
    def __init__(self, input_dim=47, reduction_ratio=4):
        super().__init__()
        hidden_dim = max(input_dim // reduction_ratio, 8)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights


class TDAE(nn.Module):
    def __init__(self, input_dim=47, latent_dim=5):
        super().__init__()
        self.attention_layer = FeatureAttention(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, latent_dim),
        )
        # Keep both training heads to validate the entire checkpoint strictly.
        self.decoder_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, input_dim),
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        calibrated, weights = self.attention_layer(x)
        latent = self.encoder(calibrated)
        return self.decoder_head(latent), self.predictor_head(latent), latent, weights


class APDFNet(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.phys_stream = nn.Linear(input_dim, 1)
        self.deep_stream = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.Mish(), nn.Linear(16, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # squeeze(-1) preserves the batch axis for single-image inference.
        alpha = self.gate(x).squeeze(-1)
        return (alpha * self.phys_stream(x).squeeze(-1)
                + (1 - alpha) * self.deep_stream(x).squeeze(-1))


def compute_glcm_texture_features(band, levels=64):
    """Training textures: positive-pixel moments, four-direction mean GLCM.

    Background and non-positive pixels remain level zero and DO participate in
    the GLCM. Removing their rows/columns would change the trained definition.
    HOM uses 1+|i-j|; ASM is sum(P**2), not sqrt(sum(P**2)).
    """
    valid = band > 0
    if valid.sum() < 100:
        raise ValueError("Each band needs at least 100 positive pixels for texture extraction.")
    pixels = band[valid]
    low, high = pixels.min(), pixels.max()
    quantized = np.zeros_like(band, dtype=np.uint8)
    if high - low > 1e-8:
        quantized[valid] = ((pixels - low) / (high - low) * (levels - 2) + 1).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels, symmetric=True, normed=True,
    )
    matrix = np.mean(glcm, axis=3).squeeze()
    i, j = np.ogrid[:levels, :levels]
    diff = np.abs(i - j)
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    mu_i, mu_j = np.sum(matrix * ii), np.sum(matrix * jj)
    sigma_i = np.sqrt(np.sum(matrix * (ii - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(matrix * (jj - mu_j) ** 2))
    if sigma_i <= 1e-8 or sigma_j <= 1e-8:
        raise ValueError("GLCM correlation is undefined for a constant/degenerate band.")
    return {
        "MEA": np.mean(pixels), "VAR": np.var(pixels),
        "HOM": np.sum(matrix / (1 + diff)),
        "CON": np.sum(matrix * diff ** 2), "DIS": np.sum(matrix * diff),
        "ENT": -np.sum(matrix * np.log2(matrix + 1e-12)),
        "ASM": np.sum(matrix ** 2),
        "Cor": np.sum(matrix * (ii - mu_i) * (jj - mu_j)) / (sigma_i * sigma_j),
    }


def training_csv_precision(features):
    """Reproduce both rounding stages of the training feature CSV export.

    Non-VAR columns: pandas round(4). Then all columns: .4e for nonzero
    magnitudes below .001, otherwise .4f. VIs are computed BEFORE rounding.
    """
    result = features.copy()
    columns = [name for name in FEATURE_NAMES if not name.startswith("VAR_")]
    result[columns] = result[columns].round(4)
    for name in FEATURE_NAMES:
        x = float(result.at[0, name])
        result.at[0, name] = float(f"{x:.4e}" if 0 < abs(x) < 0.001 else f"{x:.4f}")
    return result


def extract_image_features(image_path):
    with rasterio.open(image_path) as src:
        if src.count != 4:
            raise ValueError(f"Expected exactly four bands in order {BANDS}; got {src.count}.")
        bands = src.read()
        if not np.isfinite(bands).all():
            raise ValueError("Input contains NaN/Inf; provide the finite preprocessed TIFF.")
        if src.nodata is not None and src.nodata != 0:
            raise ValueError("Only zero-background TIFFs are supported; nonzero NoData is ambiguous.")
    # Preserve the original shared nonzero mask and float32 summation precision.
    mask = np.any(bands != 0, axis=0)
    if not mask.any():
        raise ValueError("The input image contains only background pixels.")
    means = [np.sum(band[mask]) / np.sum(mask) for band in bands]
    green, red, red_edge, nir = means

    def divide(numerator, denominator):
        if denominator == 0:
            raise ValueError("A vegetation index has a zero denominator; inference stopped.")
        return numerator / denominator

    indices = (
        divide(nir - red, nir + red), divide(nir - green, nir + green),
        divide(nir - red_edge, nir + red_edge), divide(green, red),
        divide(green, red_edge), divide(green, nir), divide(red, red_edge),
        divide(red, nir), divide(red_edge, nir), divide(nir, green),
        divide(nir - red_edge, nir + red), divide(nir - red_edge, nir - red),
        divide(nir - red_edge, red_edge - red), divide(nir, red_edge), divide(red_edge, red),
    )
    values = {}
    for band_name, band in zip(BANDS, bands):
        texture = compute_glcm_texture_features(band.astype(np.float32))
        values.update({f"{name}_{band_name}": texture[name] for name in TEXTURES})
    values.update(dict(zip(VEGETATION_INDICES, indices)))
    full_precision = pd.DataFrame([values], columns=FEATURE_NAMES, dtype=np.float64)
    if not np.isfinite(full_precision.to_numpy()).all():
        raise ValueError("Extracted features contain NaN/Inf; no imputation is performed.")
    return training_csv_precision(full_precision)


def load_scaler(path, expected_names):
    # joblib is used only with the trusted, locally supplied training artifacts.
    scaler = joblib.load(path)
    if getattr(scaler, "n_features_in_", None) != len(expected_names):
        raise ValueError(f"{path.name}: expected {len(expected_names)} input features.")
    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != list(expected_names):
            raise ValueError(f"{path.name}: feature order differs from this inference schema.")
    return scaler


def load_models(model_dir):
    """Load frozen models and scalers once for all input images."""
    model_dir = Path(model_dir).resolve()
    for path in [model_dir / name for name in MODEL_FILES]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    meta = json.loads((model_dir / "apdf_net_meta.json").read_text(encoding="utf-8"))
    if meta.get("input_dim") != 5 or meta.get("feature_cols") != LATENT_NAMES:
        raise ValueError("APDF-Net metadata must specify Latent_Feature_1 through _5 in order.")
    # CPU, one thread, eval mode and deterministic operations for reproducibility.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scaler_x = load_scaler(model_dir / "scaler_X.pkl", FEATURE_NAMES)
    scaler_apdf = load_scaler(model_dir / "apdf_net_scaler.pkl", LATENT_NAMES)
    tdae, apdf = TDAE(), APDFNet()
    tdae.load_state_dict(torch.load(model_dir / MODEL_FILES[0], map_location="cpu", weights_only=True), strict=True)
    apdf.load_state_dict(torch.load(model_dir / "apdf_net_best.pth", map_location="cpu", weights_only=True), strict=True)
    tdae.eval()
    apdf.eval()
    return tdae, apdf, scaler_x, scaler_apdf


def predict(image_path, models):
    """Return only the nitrogen estimate; intermediate values stay in memory."""
    tdae, apdf, scaler_x, scaler_apdf = models
    features = extract_image_features(image_path)
    x_scaled = scaler_x.transform(features.to_numpy())
    if not np.isfinite(x_scaled).all():
        raise ValueError("Non-finite TDAE inputs after standardization.")
    with torch.inference_mode():
        _, _, latent, _ = tdae(torch.tensor(x_scaled, dtype=torch.float32))
        latent_np = latent.numpy()
        # Training APDF-Net reads pandas' float32 latent CSV back as float64.
        # Reproduce that text round trip, rather than assuming float32->float64
        # casting is identical to the decimal values in that training CSV.
        latent_df = pd.DataFrame(latent_np, columns=LATENT_NAMES)
        latent_csv_values = pd.read_csv(io.StringIO(latent_df.to_csv(index=False))).to_numpy()
        latent_scaled = scaler_apdf.transform(latent_csv_values)
        if not np.isfinite(latent_scaled).all():
            raise ValueError("Non-finite APDF-Net inputs after standardization.")
        prediction = apdf(torch.tensor(latent_scaled, dtype=torch.float32)).item()
    if not np.isfinite(prediction):
        raise ValueError("The prediction is not finite.")
    return prediction


def find_images(image_dir):
    """Select cropped TIFFs only, in numerical plot order; no recursion."""
    pattern = re.compile(r"spectral_bg_removed_cropped_(\d+)\.tiff?", re.IGNORECASE)
    matched = [(int(match.group(1)), path) for path in image_dir.iterdir()
               if path.is_file() and (match := pattern.fullmatch(path.name))]
    matched.sort(key=lambda item: (item[0], item[1].name))
    if not matched:
        raise ValueError(f"No spectral_bg_removed_cropped_<ID>.tif images found in {image_dir}")
    if len({plot_id for plot_id, _ in matched}) != len(matched):
        raise ValueError("Duplicate plot IDs found; keep exactly one cropped TIFF per plot.")
    return [path for _, path in matched]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "args.yaml",
                        help="YAML path configuration; command-line paths override it.")
    parser.add_argument("--image", type=Path, default=None,
                        help="A four-band TIFF or a folder containing cropped TIFFs.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or set(config) - {"image", "model_dir", "output_dir"}:
        raise ValueError("Config must be a mapping with image, model_dir and output_dir paths only.")

    def configured_path(key, default):
        value = config.get(key)
        if value is None:
            return default
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config {key} must be a nonempty path string or null.")
        path = Path(value)
        return (config_path.parent / path).resolve()

    image_path = (args.image or configured_path("image", DEFAULT_IMAGE)).resolve()
    model_dir = args.model_dir or configured_path("model_dir", BASE_DIR)
    configured_output = args.output_dir or configured_path("output_dir", None)
    if not image_path.exists():
        raise FileNotFoundError(f"Input not found: {image_path}")
    if image_path.is_dir():
        images = find_images(image_path)
        output_dir = configured_output or BASE_DIR / "inference_results" / image_path.name
    else:
        images = [image_path]
        output_dir = configured_output or BASE_DIR / "inference_results" / image_path.parent.name / image_path.stem
    models = load_models(model_dir)
    rows = []
    for index, image in enumerate(images, start=1):
        try:
            prediction = predict(image, models)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"{image.name}: {exc}") from exc
        formatted_prediction = f"{prediction:.3f}%"
        rows.append({"Image Name": image.name, "predicted_N_content": formatted_prediction})
        print(f"[{index}/{len(images)}] {image.name}: {formatted_prediction}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "predictions_summary.json"
    result_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"Results (original training target unit): {result_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)
