#!/usr/bin/env python
"""Clean-room Python approximation of the PLUS LEAS/CARS workflow.

This file is based on the public PLUS paper, manuals, and sample parameter
files in this directory. It does not extract or reuse code from the PLUS exe.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:  # pragma: no cover - handled at runtime
    rasterio = None
    Resampling = None
    reproject = None


NODATA_EXPANSION = 255


def require_rasterio() -> None:
    if rasterio is None:
        raise SystemExit(
            "This tool needs rasterio. On this machine use the system Python "
            "where `python -c \"import rasterio\"` succeeds."
        )


def as_path(value: str | Path) -> Path:
    return Path(str(value).strip().strip('"'))


def parse_tagged_file(path: str | Path) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    current: str | None = None
    for raw in as_path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            current = line[1:-1]
            tags.setdefault(current, [])
        elif current is not None:
            tags[current].append(line)
    return tags


def first_tag(tags: dict[str, list[str]], names: Iterable[str], default: str | None = None) -> str | None:
    for name in names:
        values = tags.get(name)
        if values:
            return values[0]
    return default


def parse_csv_floats(line: str) -> list[float]:
    return [float(x.strip()) for x in line.split(",") if x.strip()]


def parse_csv_ints(line: str) -> list[int]:
    return [int(float(x.strip())) for x in line.split(",") if x.strip()]


def parse_classes(value: str | None, count: int | None = None) -> list[int]:
    if value:
        return parse_csv_ints(value)
    if count is None:
        raise ValueError("Class count is required when --classes is omitted.")
    return list(range(1, count + 1))


def same_grid(src, ref) -> bool:
    return (
        src.width == ref.width
        and src.height == ref.height
        and src.transform == ref.transform
        and src.crs == ref.crs
    )


def valid_mask(arr: np.ndarray, nodata) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None:
        try:
            nd = float(nodata)
        except (TypeError, ValueError, OverflowError):
            return mask
        if not math.isfinite(nd):
            return mask
        if np.issubdtype(arr.dtype, np.floating):
            info = np.finfo(arr.dtype)
            if nd < float(info.min) or nd > float(info.max):
                return mask
        mask &= arr != nd
    return mask


def read_aligned(path: str | Path, ref, *, resampling_name: str = "bilinear") -> tuple[np.ndarray, np.ndarray]:
    require_rasterio()
    path = as_path(path)
    with rasterio.open(path) as src:
        nodata = src.nodata
        if same_grid(src, ref):
            with np.errstate(over="ignore", invalid="ignore"):
                arr = src.read(1).astype("float32", copy=False)
        else:
            arr = np.full((ref.height, ref.width), np.nan, dtype="float32")
            resampling = Resampling.nearest if resampling_name == "nearest" else Resampling.bilinear
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return arr, valid_mask(arr, nodata)


def write_raster(path: str | Path, arr: np.ndarray, ref_profile: dict, *, nodata=None, dtype=None) -> None:
    require_rasterio()
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype or arr.dtype,
        nodata=nodata,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(profile["dtype"], copy=False), 1)


def write_multiband(path: str | Path, bands: list[np.ndarray], ref_profile: dict, *, nodata=None, dtype="uint8") -> None:
    require_rasterio()
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=len(bands),
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, band in enumerate(bands, 1):
            dst.write(band.astype(dtype, copy=False), i)


def ensure_writable(path: str | Path, overwrite: bool) -> None:
    path = as_path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path}. Add --overwrite to replace it.")


def normalize_factor(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    out = np.zeros(arr.shape, dtype="float32")
    if not np.any(mask):
        return out, (0.0, 0.0)
    values = arr[mask]
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return out, (lo, hi)
    out[mask] = (arr[mask] - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out, (lo, hi)


def tif_files(folder: str | Path) -> list[Path]:
    files = sorted(as_path(folder).glob("*.tif"))
    if not files:
        raise SystemExit(f"No .tif files found in {folder}")
    return files


def extract_expansion(args: argparse.Namespace) -> None:
    require_rasterio()
    ensure_writable(args.out, args.overwrite)
    with rasterio.open(args.start) as start_ds, rasterio.open(args.end) as end_ds:
        if not same_grid(start_ds, end_ds):
            raise SystemExit("Start and end LULC rasters must share the same grid for extract-expansion.")
        start = start_ds.read(1)
        end = end_ds.read(1)
        start_valid = valid_mask(start.astype("float32"), start_ds.nodata)
        end_valid = valid_mask(end.astype("float32"), end_ds.nodata)
        valid = start_valid & end_valid
        out = np.full(start.shape, NODATA_EXPANSION, dtype="uint8")
        out[valid] = 0
        changed = valid & (start != end)
        out[changed] = end[changed].astype("uint8")
        write_raster(args.out, out, start_ds.profile, nodata=NODATA_EXPANSION, dtype="uint8")
    print(f"Wrote expansion map: {args.out}")


@dataclass
class TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    gain: float = 0.0


class RandomTreeRegressor:
    def __init__(
        self,
        *,
        max_depth: int,
        min_leaf: int,
        max_features: int,
        n_thresholds: int,
        rng: np.random.Generator,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng
        self.nodes: list[TreeNode] = []
        self.feature_gains: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomTreeRegressor":
        self.nodes = []
        self.feature_gains = np.zeros(x.shape[1], dtype="float64")
        self._build(x, y.astype("float32", copy=False), np.arange(x.shape[0]), 0)
        return self

    def _sse(self, y: np.ndarray, idx: np.ndarray) -> float:
        n = idx.size
        if n == 0:
            return 0.0
        vals = y[idx]
        total = float(vals.sum())
        total_sq = float(np.dot(vals, vals))
        return total_sq - (total * total / n)

    def _build(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> int:
        value = float(y[idx].mean()) if idx.size else 0.0
        node_id = len(self.nodes)
        self.nodes.append(TreeNode(value=value))

        if depth >= self.max_depth or idx.size < self.min_leaf * 2:
            return node_id
        if value <= 1e-7 or value >= 1.0 - 1e-7:
            return node_id

        parent_sse = self._sse(y, idx)
        if parent_sse <= 1e-12:
            return node_id

        feature_count = x.shape[1]
        max_features = min(max(1, self.max_features), feature_count)
        features = self.rng.choice(feature_count, size=max_features, replace=False)

        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        sample_idx = idx
        if idx.size > 4096:
            sample_idx = self.rng.choice(idx, size=4096, replace=False)

        for feature in features:
            sample_values = x[sample_idx, feature]
            lo = float(np.min(sample_values))
            hi = float(np.max(sample_values))
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                continue
            quantiles = self.rng.uniform(0.05, 0.95, size=self.n_thresholds)
            thresholds = np.unique(np.quantile(sample_values, quantiles))
            values = x[idx, feature]
            for threshold in thresholds:
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = idx.size - n_left
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]
                child_sse = self._sse(y, left_idx) + self._sse(y, right_idx)
                gain = parent_sse - child_sse
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_idx
                    best_right = right_idx

        if best_feature < 0 or best_left is None or best_right is None:
            return node_id

        left_id = self._build(x, y, best_left, depth + 1)
        right_id = self._build(x, y, best_right, depth + 1)
        self.nodes[node_id] = TreeNode(
            value=value,
            feature=best_feature,
            threshold=best_threshold,
            left=left_id,
            right=right_id,
            gain=best_gain,
        )
        assert self.feature_gains is not None
        self.feature_gains[best_feature] += best_gain
        return node_id

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape[0], dtype="float32")
        stack: list[tuple[np.ndarray, int]] = [(np.arange(x.shape[0]), 0)]
        while stack:
            idx, node_id = stack.pop()
            if idx.size == 0:
                continue
            node = self.nodes[node_id]
            if node.feature < 0:
                out[idx] = node.value
                continue
            mask = x[idx, node.feature] <= node.threshold
            stack.append((idx[mask], node.left))
            stack.append((idx[~mask], node.right))
        return out


class SimpleRandomForestRegressor:
    def __init__(
        self,
        *,
        n_trees: int,
        max_depth: int,
        min_leaf: int,
        max_features: int,
        n_thresholds: int,
        seed: int,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.seed = seed
        self.trees: list[RandomTreeRegressor] = []
        self.feature_importances_: np.ndarray | None = None
        self.oob_rmse_: float | None = None
        self.train_rmse_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleRandomForestRegressor":
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        oob_sum = np.zeros(n, dtype="float64")
        oob_count = np.zeros(n, dtype="uint16")
        importances = np.zeros(x.shape[1], dtype="float64")
        self.trees = []

        for _ in range(self.n_trees):
            boot = rng.integers(0, n, size=n)
            tree = RandomTreeRegressor(
                max_depth=self.max_depth,
                min_leaf=self.min_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                rng=np.random.default_rng(int(rng.integers(1, 2**31 - 1))),
            )
            tree.fit(x[boot], y[boot])
            self.trees.append(tree)
            if tree.feature_gains is not None:
                importances += tree.feature_gains

            seen = np.zeros(n, dtype=bool)
            seen[np.unique(boot)] = True
            oob_idx = np.flatnonzero(~seen)
            if oob_idx.size:
                oob_sum[oob_idx] += tree.predict(x[oob_idx])
                oob_count[oob_idx] += 1

        train_pred = self.predict(x)
        self.train_rmse_ = rmse(y, train_pred)
        has_oob = oob_count > 0
        if np.any(has_oob):
            self.oob_rmse_ = rmse(y[has_oob], (oob_sum[has_oob] / oob_count[has_oob]).astype("float32"))
        total = float(importances.sum())
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.zeros(x.shape[0], dtype="float32")
        for tree in self.trees:
            pred += tree.predict(x)
        if self.trees:
            pred /= len(self.trees)
        return np.clip(pred, 0.0, 1.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.astype("float32") - y_pred.astype("float32")) ** 2)))


def sample_indices(
    valid_flat: np.ndarray,
    labels_flat: np.ndarray,
    class_id: int,
    *,
    sampling_rate: float,
    max_samples: int,
    balanced: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if balanced:
        pos = valid_flat[labels_flat[valid_flat] == class_id]
        neg = valid_flat[labels_flat[valid_flat] != class_id]
        if pos.size == 0:
            raise SystemExit(f"Class {class_id} has no positive expansion samples.")
        per_group = min(max_samples // 2, pos.size, neg.size)
        per_group = max(per_group, min(pos.size, neg.size, 1))
        return np.concatenate(
            [
                rng.choice(pos, size=per_group, replace=pos.size < per_group),
                rng.choice(neg, size=per_group, replace=neg.size < per_group),
            ]
        )
    n = min(max_samples, max(1, int(valid_flat.size * sampling_rate)))
    return rng.choice(valid_flat, size=n, replace=valid_flat.size < n)


def make_x_from_flat(factors: list[np.ndarray], flat_idx: np.ndarray) -> np.ndarray:
    return np.column_stack([factor.ravel()[flat_idx] for factor in factors]).astype("float32", copy=False)


def train_leas(args: argparse.Namespace) -> None:
    require_rasterio()
    if args.params:
        tags = parse_tagged_file(args.params)
        args.expansion = args.expansion or first_tag(tags, ["Input LULC"])
        args.factors_dir = args.factors_dir or first_tag(tags, ["Input Featrue folder", "Input Feature folder"])
        args.out_prefix = args.out_prefix or first_tag(tags, ["Output probability"])
        args.sampling_rate = args.sampling_rate if args.sampling_rate is not None else float(first_tag(tags, ["Input sampling rate"], "0.01"))
        args.mtry = args.mtry or int(float(first_tag(tags, ["mTry"], "0") or 0))
        args.trees = args.trees or int(float(first_tag(tags, ["Input the number of trees"], "20") or 20))
        args.balanced = args.balanced or bool(int(float(first_tag(tags, ["Is balance?"], "0") or 0)))

    if not args.expansion or not args.factors_dir or not args.out_prefix:
        raise SystemExit("train-leas needs --expansion, --factors-dir, and --out-prefix, or --params.")

    rng = np.random.default_rng(args.seed)
    with rasterio.open(args.expansion) as exp_ds:
        expansion = exp_ds.read(1)
        exp_valid = valid_mask(expansion.astype("float32"), exp_ds.nodata) & (expansion != NODATA_EXPANSION)
        factor_paths = tif_files(args.factors_dir)
        factors: list[np.ndarray] = []
        factor_valid = np.ones(expansion.shape, dtype=bool)
        minmax_rows: list[tuple[str, float, float]] = []
        for path in factor_paths:
            arr, mask = read_aligned(path, exp_ds, resampling_name="bilinear")
            norm, (lo, hi) = normalize_factor(arr, mask)
            factors.append(norm)
            factor_valid &= mask
            minmax_rows.append((path.name, lo, hi))

        valid = exp_valid & factor_valid
        valid_flat = np.flatnonzero(valid.ravel())
        if valid_flat.size == 0:
            raise SystemExit("No valid pixels after aligning expansion raster and factors.")

        classes = parse_classes(args.classes, None) if args.classes else sorted(int(v) for v in np.unique(expansion[valid]) if int(v) > 0)
        if not classes:
            raise SystemExit("No positive expansion classes found.")

        out_prefix = as_path(args.out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        ensure_writable(out_prefix, args.overwrite)
        for class_id in classes:
            ensure_writable(out_prefix.with_name(f"{out_prefix.stem}_band_{class_id}.tif"), args.overwrite)
        bands: list[np.ndarray] = []
        labels_flat = expansion.ravel()

        max_features = args.mtry or len(factors)
        max_features = min(max_features, len(factors))
        for class_pos, class_id in enumerate(classes, 1):
            print(f"Training LEAS class {class_id} ({class_pos}/{len(classes)})")
            flat_idx = sample_indices(
                valid_flat,
                labels_flat,
                class_id,
                sampling_rate=args.sampling_rate,
                max_samples=args.max_samples,
                balanced=args.balanced,
                rng=rng,
            )
            x_train = make_x_from_flat(factors, flat_idx)
            y_train = (labels_flat[flat_idx] == class_id).astype("float32")
            if y_train.sum() == 0:
                print(f"  skipped class {class_id}: no positives in sampled data")
                bands.append(np.zeros(expansion.shape, dtype="uint8"))
                continue

            forest = SimpleRandomForestRegressor(
                n_trees=args.trees,
                max_depth=args.max_depth,
                min_leaf=args.min_leaf,
                max_features=max_features,
                n_thresholds=args.thresholds,
                seed=int(rng.integers(1, 2**31 - 1)),
            ).fit(x_train, y_train)

            prob_flat = np.zeros(expansion.size, dtype="uint8")
            for start in range(0, valid_flat.size, args.prediction_chunk_size):
                chunk_idx = valid_flat[start : start + args.prediction_chunk_size]
                x_chunk = make_x_from_flat(factors, chunk_idx)
                pred = forest.predict(x_chunk)
                prob_flat[chunk_idx] = np.rint(np.clip(pred, 0.0, 1.0) * 255.0).astype("uint8")

            prob = prob_flat.reshape(expansion.shape)
            band_path = out_prefix.with_name(f"{out_prefix.stem}_band_{class_id}.tif")
            write_raster(band_path, prob, exp_ds.profile, nodata=0, dtype="uint8")
            bands.append(prob)
            print(
                f"  wrote {band_path.name}; RMSE={forest.train_rmse_:.6f}, "
                f"OOB_RMSE={(forest.oob_rmse_ if forest.oob_rmse_ is not None else float('nan')):.6f}"
            )
            write_contribution_csv(
                out_prefix.parent / f"Contribution_class_{class_id}.csv",
                factor_paths,
                forest,
                x_train,
                y_train,
            )

        write_multiband(out_prefix, bands, exp_ds.profile, nodata=0, dtype="uint8")
        with (out_prefix.parent / "imageminmax_openplus.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["factor", "min", "max"])
            writer.writerows(minmax_rows)
    print(f"Wrote multiband probability raster: {out_prefix}")


def write_contribution_csv(
    path: Path,
    factor_paths: list[Path],
    forest: SimpleRandomForestRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    base_pred = forest.predict(x_train)
    base_rmse = rmse(y_train, base_pred)
    rng = np.random.default_rng(12345)
    noise_rmses: list[float] = []
    for feature in range(x_train.shape[1]):
        x_noise = x_train.copy()
        rng.shuffle(x_noise[:, feature])
        noise_rmses.append(rmse(y_train, forest.predict(x_noise)))
    raw = np.maximum(np.array(noise_rmses) - base_rmse, 0.0)
    total = float(raw.sum())
    contribution = raw / total if total > 0 else np.zeros_like(raw)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["RMSE_original", base_rmse])
        writer.writerow(["Factors", *[p.name for p in factor_paths]])
        writer.writerow(["RMSE_noise", *noise_rmses])
        writer.writerow(["Contribution", *contribution.tolist()])


def box_count(mask: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(mask.astype("uint8"), pad_width=pad, mode="constant", constant_values=0)
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")
    counts = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return counts.astype("float32") - mask.astype("float32")


def load_probability_layers(paths: list[str], ref, count: int) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            if src.count >= count:
                for i in range(1, count + 1):
                    if same_grid(src, ref):
                        arr = src.read(i).astype("float32")
                    else:
                        arr = np.full((ref.height, ref.width), np.nan, dtype="float32")
                        reproject(
                            source=rasterio.band(src, i),
                            destination=arr,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_transform=ref.transform,
                            dst_crs=ref.crs,
                            dst_nodata=np.nan,
                            resampling=Resampling.bilinear,
                        )
                    layers.append(probability_to_unit(arr))
                return layers
    for path in paths:
        arr, _ = read_aligned(path, ref, resampling_name="bilinear")
        layers.append(probability_to_unit(arr))
    if len(layers) != count:
        raise SystemExit(f"Expected {count} probability layers, got {len(layers)}.")
    return layers


def probability_to_unit(arr: np.ndarray) -> np.ndarray:
    out = arr.astype("float32", copy=True)
    finite = np.isfinite(out)
    if np.any(finite) and float(np.nanmax(out[finite])) > 1.5:
        out /= 255.0
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def parse_cars_params(args: argparse.Namespace) -> argparse.Namespace:
    if not args.params:
        return args
    tags = parse_tagged_file(args.params)
    args.initial = args.initial or first_tag(tags, ["Input LULC"])
    args.out = args.out or first_tag(tags, ["Output simulation"])
    args.policy = args.policy or first_tag(tags, ["Input Policy"])
    args.classes = args.classes or first_tag(tags, ["Input classes"])
    if args.classes and "," not in args.classes:
        args.classes = ",".join(str(i) for i in range(1, int(float(args.classes)) + 1))
    args.probabilities = args.probabilities or tags.get("Input Probability Folder", [])
    args.neighborhood = args.neighborhood or int(float(first_tag(tags, ["Input Neighborhood"], "3")))
    args.years = args.years or int(float(first_tag(tags, ["How many years"], "1")))
    args.patch_generation = (
        args.patch_generation
        if args.patch_generation is not None
        else float(first_tag(tags, ["Patch generation"], "0.5"))
    )
    args.expansion_coefficient = (
        args.expansion_coefficient
        if args.expansion_coefficient is not None
        else float(first_tag(tags, ["Expansion coefficient"], "0.1"))
    )
    args.seed_percentage = (
        args.seed_percentage
        if args.seed_percentage is not None
        else float(first_tag(tags, ["Percentage of seeds"], "0.1"))
    )
    args.weights = args.weights or first_tag(tags, ["Neighborhood Weight"])
    if args.transition_matrix is None and tags.get("Transition matrix"):
        args.transition_matrix = ";".join(tags["Transition matrix"])
    if args.demands is None and tags.get("Years and corresponding demands"):
        args.demands = tags["Years and corresponding demands"][0]
    return args


def transition_allowed(current: np.ndarray, target: int, classes: list[int], matrix: np.ndarray) -> np.ndarray:
    out = np.zeros(current.shape, dtype=bool)
    target_idx = classes.index(target)
    for current_idx, current_class in enumerate(classes):
        # Bundled PLUS parameter files behave as rows=current classes and
        # columns=future classes. This orientation also makes the sample demand
        # vector reachable; using the opposite orientation locks class 4.
        if matrix[current_idx, target_idx] == 1:
            out |= current == current_class
    return out


def counts_for_classes(arr: np.ndarray, valid: np.ndarray, classes: list[int]) -> np.ndarray:
    return np.array([int(np.sum(valid & (arr == class_id))) for class_id in classes], dtype="int64")


def classification_metrics(truth: np.ndarray, simulation: np.ndarray, valid: np.ndarray, classes: list[int]) -> tuple[float, float]:
    total = int(valid.sum())
    if total == 0:
        return float("nan"), float("nan")
    correct = int(np.sum(valid & (truth == simulation)))
    overall = correct / total
    row_counts = np.array([int(np.sum(valid & (truth == class_id))) for class_id in classes], dtype="float64")
    col_counts = np.array([int(np.sum(valid & (simulation == class_id))) for class_id in classes], dtype="float64")
    expected = float(np.dot(row_counts, col_counts)) / (total * total)
    kappa = (overall - expected) / (1.0 - expected) if expected < 1.0 else float("nan")
    return overall, kappa


def simulate_cars(args: argparse.Namespace) -> None:
    require_rasterio()
    args = parse_cars_params(args)
    if not args.initial or not args.out or not args.probabilities:
        raise SystemExit("simulate-cars needs --initial, --probabilities, and --out, or --params.")
    if args.neighborhood % 2 != 1 or args.neighborhood < 3:
        raise SystemExit("--neighborhood must be an odd integer >= 3.")

    classes = parse_classes(args.classes, len(args.probabilities))
    weights = np.array(parse_csv_floats(args.weights), dtype="float32") if args.weights else np.ones(len(classes), dtype="float32")
    if weights.size != len(classes):
        raise SystemExit("Neighborhood weight count must equal class count.")
    transition = parse_transition(args.transition_matrix, len(classes))
    demand_values = parse_demands(args.demands, len(classes))
    ensure_writable(args.out, args.overwrite)
    if args.history_csv:
        ensure_writable(args.history_csv, args.overwrite)
    if args.metrics_csv:
        ensure_writable(args.metrics_csv, args.overwrite)

    rng = np.random.default_rng(args.seed)
    with rasterio.open(args.initial) as init_ds:
        current = init_ds.read(1).copy()
        valid = valid_mask(current.astype("float32"), init_ds.nodata)
        if args.policy:
            policy, policy_valid = read_aligned(args.policy, init_ds, resampling_name="nearest")
            convertible = policy_valid & (policy == 1)
        else:
            convertible = np.ones(current.shape, dtype=bool)
        convertible &= valid
        probabilities = load_probability_layers(args.probabilities, init_ds, len(classes))
        truth = None
        metric_valid = None
        best_current = None
        best_metric_score = -float("inf")
        best_iteration = 0
        best_overall = float("nan")
        best_kappa = float("nan")
        metrics_rows: list[list[object]] = []
        if args.truth:
            with rasterio.open(args.truth) as truth_ds:
                if not same_grid(init_ds, truth_ds):
                    raise SystemExit("--truth raster must share the initial raster grid.")
                truth = truth_ds.read(1)
                metric_valid = valid & valid_mask(truth.astype("float32"), truth_ds.nodata)
            overall, kappa = classification_metrics(truth, current, metric_valid, classes)
            metric_value = kappa if args.select_best_by == "kappa" else overall
            best_metric_score = metric_value
            best_iteration = 0
            best_overall = overall
            best_kappa = kappa
            best_current = current.copy()
            metrics_rows.append([0, overall, kappa, *counts_for_classes(current, valid, classes).tolist()])
        max_iterations = args.max_iterations or max(50, int(args.years) * 100)
        history: list[list[int]] = []
        inertia = np.ones(len(classes), dtype="float32")
        previous_gap = demand_values - counts_for_classes(current, valid, classes)

        for iteration in range(1, max_iterations + 1):
            counts = counts_for_classes(current, valid, classes)
            gap = demand_values - counts
            history.append([iteration, *counts.tolist()])
            if np.all(np.abs(gap) <= args.tolerance):
                print(f"CARS stopped at iteration {iteration}: demands reached within tolerance.")
                break

            positive = np.flatnonzero(gap > args.tolerance)
            if positive.size == 0:
                print(f"CARS stopped at iteration {iteration}: no class has positive demand gap.")
                break

            for idx in positive:
                if abs(previous_gap[idx]) > 0 and abs(gap[idx]) > abs(previous_gap[idx]):
                    inertia[idx] *= min(2.0, abs(gap[idx]) / max(abs(previous_gap[idx]), 1))
                else:
                    inertia[idx] = max(0.25, inertia[idx] * 0.98)
            previous_gap = gap.copy()

            best_cell_score = np.full(current.shape, -1.0, dtype="float32")
            best_class = np.zeros(current.shape, dtype=current.dtype)
            remaining_iterations = max(1, max_iterations - iteration + 1)

            for idx in positive:
                class_id = classes[idx]
                class_mask = current == class_id
                neighbor = box_count(class_mask, args.neighborhood) / float(args.neighborhood * args.neighborhood - 1)
                neighbor *= weights[idx]
                prob = probabilities[idx]
                random_values = rng.random(current.shape, dtype="float32")
                seed_mask = (neighbor <= 0) & (random_values < (prob * args.seed_percentage))
                seed_bonus = seed_mask.astype("float32") * args.expansion_coefficient * random_values
                score = prob * (neighbor + seed_bonus) * inertia[idx]
                allowed = transition_allowed(current, class_id, classes, transition)
                score[~(allowed & convertible & valid)] = -1.0
                score[current == class_id] = -1.0
                replace = score > best_cell_score
                best_cell_score[replace] = score[replace]
                best_class[replace] = class_id

            changed_total = 0
            gate = rng.random(current.shape, dtype="float32") < np.clip(
                best_cell_score / max(args.patch_generation, 1e-6), 0.0, 1.0
            )
            for idx in positive:
                class_id = classes[idx]
                candidates = (best_class == class_id) & (best_cell_score > 0)
                gated = candidates & gate
                quota = int(math.ceil(max(0, gap[idx]) / remaining_iterations))
                quota = max(1, min(int(gap[idx]), quota))
                chosen = choose_top(best_cell_score, gated, quota)
                if chosen.size < quota:
                    fallback = candidates.copy()
                    if chosen.size:
                        fallback.ravel()[chosen] = False
                    extra = choose_top(best_cell_score, fallback, quota - chosen.size)
                    chosen = np.concatenate([chosen, extra])
                if chosen.size:
                    current.ravel()[chosen] = class_id
                    changed_total += chosen.size

            print(f"Iteration {iteration}: changed {changed_total} cells; gaps {gap.tolist()}")
            if truth is not None and metric_valid is not None:
                overall, kappa = classification_metrics(truth, current, metric_valid, classes)
                metric_value = kappa if args.select_best_by == "kappa" else overall
                metrics_rows.append([iteration, overall, kappa, *counts_for_classes(current, valid, classes).tolist()])
                if metric_value > best_metric_score:
                    best_metric_score = metric_value
                    best_iteration = iteration
                    best_overall = overall
                    best_kappa = kappa
                    best_current = current.copy()
            if changed_total == 0:
                print("CARS stopped: no convertible candidates remain.")
                break

        if best_current is not None:
            current = best_current
            print(
                f"Selected iteration {best_iteration} by {args.select_best_by}: "
                f"overall_accuracy={best_overall:.6f}, kappa={best_kappa:.6f}"
            )
        out = current.astype(init_ds.dtypes[0], copy=False)
        write_raster(args.out, out, init_ds.profile, nodata=init_ds.nodata, dtype=init_ds.dtypes[0])
        if args.history_csv:
            write_history(args.history_csv, history, classes)
        if args.metrics_csv and metrics_rows:
            write_metrics(args.metrics_csv, metrics_rows, classes)
    print(f"Wrote simulation raster: {args.out}")


def choose_top(score: np.ndarray, mask: np.ndarray, quota: int) -> np.ndarray:
    flat = np.flatnonzero(mask.ravel())
    if flat.size == 0 or quota <= 0:
        return np.array([], dtype=np.int64)
    if flat.size <= quota:
        return flat
    values = score.ravel()[flat]
    selected = np.argpartition(values, -quota)[-quota:]
    return flat[selected]


def parse_transition(value: str | None, count: int) -> np.ndarray:
    if not value:
        return np.ones((count, count), dtype="uint8")
    rows = [parse_csv_ints(row) for row in value.split(";") if row.strip()]
    matrix = np.array(rows, dtype="uint8")
    if matrix.shape != (count, count):
        raise SystemExit(f"Transition matrix must be {count}x{count}; got {matrix.shape}.")
    return matrix


def parse_demands(value: str | None, count: int) -> np.ndarray:
    if not value:
        raise SystemExit("--demands is required unless --params provides it.")
    nums = parse_csv_ints(value)
    if len(nums) == count + 1:
        nums = nums[1:]
    if len(nums) != count:
        raise SystemExit(f"Demand count must be {count}; got {len(nums)}.")
    return np.array(nums, dtype="int64")


def write_history(path: str | Path, history: list[list[int]], classes: list[int]) -> None:
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", *[f"class_{c}" for c in classes]])
        writer.writerows(history)


def write_metrics(path: str | Path, rows: list[list[object]], classes: list[int]) -> None:
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "overall_accuracy", "kappa", *[f"class_{c}" for c in classes]])
        writer.writerows(rows)


def validate(args: argparse.Namespace) -> None:
    require_rasterio()
    with rasterio.open(args.truth) as truth_ds, rasterio.open(args.simulation) as sim_ds:
        if not same_grid(truth_ds, sim_ds):
            raise SystemExit("Truth and simulation rasters must share a grid.")
        truth = truth_ds.read(1)
        sim = sim_ds.read(1)
        valid = valid_mask(truth.astype("float32"), truth_ds.nodata) & valid_mask(sim.astype("float32"), sim_ds.nodata)
        if args.initial:
            with rasterio.open(args.initial) as init_ds:
                if not same_grid(truth_ds, init_ds):
                    raise SystemExit("Initial raster must share the truth grid.")
                initial = init_ds.read(1)
                valid &= valid_mask(initial.astype("float32"), init_ds.nodata)
        else:
            initial = None

    classes = sorted(int(v) for v in np.unique(np.concatenate([truth[valid], sim[valid]])))
    matrix = np.zeros((len(classes), len(classes)), dtype="int64")
    class_to_idx = {value: i for i, value in enumerate(classes)}
    for t, s in zip(truth[valid], sim[valid]):
        matrix[class_to_idx[int(t)], class_to_idx[int(s)]] += 1

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall = correct / total if total else float("nan")
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    expected = float(np.dot(row_sum, col_sum)) / (total * total) if total else float("nan")
    kappa = (overall - expected) / (1.0 - expected) if total and expected < 1.0 else float("nan")

    rows: list[list[object]] = []
    rows.append(["overall_accuracy", overall])
    rows.append(["kappa", kappa])
    rows.append([])
    rows.append(["truth\\simulation", *classes])
    for class_id, row in zip(classes, matrix):
        rows.append([class_id, *row.tolist()])

    if initial is not None:
        observed_change = valid & (truth != initial)
        predicted_change = valid & (sim != initial)
        hits = observed_change & predicted_change & (truth == sim)
        misses = observed_change & ~predicted_change
        false_alarm = ~observed_change & predicted_change
        wrong_hits = observed_change & predicted_change & (truth != sim)
        denom = int(hits.sum() + misses.sum() + false_alarm.sum() + wrong_hits.sum())
        fom = int(hits.sum()) / denom if denom else float("nan")
        rows.append([])
        rows.extend(
            [
                ["figure_of_merit", fom],
                ["hits", int(hits.sum())],
                ["misses", int(misses.sum())],
                ["false_alarms", int(false_alarm.sum())],
                ["wrong_hits", int(wrong_hits.sum())],
            ]
        )

    if args.out_csv:
        ensure_writable(args.out_csv, args.overwrite)
        out_csv = as_path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"Wrote validation CSV: {out_csv}")
    else:
        for row in rows:
            print(",".join(str(x) for x in row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open Python approximation of the PLUS LEAS/CARS workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-expansion", help="Create a PLUS-style expansion map from start/end LULC rasters.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=extract_expansion)

    p = sub.add_parser("train-leas", help="Train clean-room LEAS probability surfaces from expansion and factors.")
    p.add_argument("--params", help="PLUS LEASparameters.tmp")
    p.add_argument("--expansion")
    p.add_argument("--factors-dir")
    p.add_argument("--out-prefix")
    p.add_argument("--classes", help="Comma-separated target classes; default uses positive classes in expansion map.")
    p.add_argument("--sampling-rate", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--trees", type=int)
    p.add_argument("--mtry", type=int)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-leaf", type=int, default=30)
    p.add_argument("--thresholds", type=int, default=8)
    p.add_argument("--prediction-chunk-size", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=train_leas)

    p = sub.add_parser("simulate-cars", help="Run a clean-room CARS cellular automata simulation.")
    p.add_argument("--params", help="PLUS CARSparameters.tmp")
    p.add_argument("--initial")
    p.add_argument("--probabilities", nargs="*", help="Probability band files, or one multiband probability raster.")
    p.add_argument("--out")
    p.add_argument("--policy", help="Binary convertible raster: 1 convertible, 0 locked.")
    p.add_argument("--classes", help="Comma-separated class ids; default is 1..number of probability layers.")
    p.add_argument("--demands", help="Comma-separated demands, optionally prefixed by year index.")
    p.add_argument("--weights", help="Comma-separated neighborhood weights.")
    p.add_argument("--transition-matrix", help="Rows separated by ';', values by ','. Rows are current classes, columns are future classes.")
    p.add_argument("--neighborhood", type=int)
    p.add_argument("--years", type=int)
    p.add_argument("--patch-generation", type=float, default=None)
    p.add_argument("--expansion-coefficient", type=float, default=None)
    p.add_argument("--seed-percentage", type=float, default=None)
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--tolerance", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-csv")
    p.add_argument("--truth", help="Optional truth raster used to save the best iteration by OA or Kappa.")
    p.add_argument("--select-best-by", choices=["kappa", "overall_accuracy"], default="kappa")
    p.add_argument("--metrics-csv", help="Optional per-iteration OA/Kappa metrics CSV when --truth is set.")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=simulate_cars)

    p = sub.add_parser("validate", help="Calculate confusion-matrix, kappa, and optional FoM metrics.")
    p.add_argument("--truth", required=True)
    p.add_argument("--simulation", required=True)
    p.add_argument("--initial", help="Initial map for FoM statistics.")
    p.add_argument("--out-csv")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
