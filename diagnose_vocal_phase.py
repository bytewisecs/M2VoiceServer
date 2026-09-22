"""只读统计训练/验证集的相位尺度；仅依赖 NumPy，不启动训练或修改数据。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from mDataloader.exclusions import INVALID_GROUPS
from remove_abnormal_samples import SAMPLE_IDS


NAME_PATTERN = re.compile(r"^mmW_(\d{8}_\d{6})_Vib_(s(?:10|[1-9]))\.npy$", re.IGNORECASE)


def phase_statistics(raw):
    values = np.asarray(raw).squeeze()
    if values.ndim != 2 or 256 not in values.shape:
        raise ValueError(f"期望 [T,256] 或 [256,T]，实际 {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("原始特征含 NaN/Inf")
    ambiguous = values.shape == (256, 256)
    # 与 trainv4 的方向约定一致；方阵需人工核实时间轴。
    if values.shape[1] != 256:
        values = values.T
    if values.shape[0] < 2:
        raise ValueError("时间维不足 2 帧")
    complex_input = np.iscomplexobj(values)
    phase = np.angle(values.astype(np.complex128)) if complex_input else values.astype(np.float64)
    unit = np.exp(1j * phase)
    temporal_residual = unit - unit.mean(axis=0, keepdims=True)
    channel_rms = np.sqrt(np.mean(np.abs(temporal_residual) ** 2, axis=0))
    step = np.angle(unit[1:] * np.conj(unit[:-1]))
    return {
        "input_kind": "complex" if complex_input else "real_assumed_radians",
        "frames": int(values.shape[0]),
        "ambiguous_time_axis": ambiguous,
        "raw_zero_fraction": float(np.mean(values == 0)),
        "phase_std_radians": float(np.std(phase)),
        "real_outside_pi_fraction": float(np.mean(np.abs(phase) > np.pi)) if not complex_input else 0.0,
        "circular_temporal_rms": float(np.sqrt(np.mean(channel_rms ** 2))),
        "channel_temporal_rms_median": float(np.median(channel_rms)),
        "adjacent_phase_step_rms_radians": float(np.sqrt(np.mean(step ** 2))),
    }


def summarize_split(split_root, split, noise_std):
    folder = split_root / split / "mmVocal"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows, errors = [], []
    excluded = 0
    for path in sorted(folder.glob("*.npy")):
        match = NAME_PATTERN.fullmatch(path.name)
        if match is None:
            errors.append({"file": path.name, "error": "文件名不符合训练命名约定"})
            continue
        group, suffix = match.groups()
        sample_id = f"audio_{group}_{suffix.lower()}"
        if group in INVALID_GROUPS or sample_id in SAMPLE_IDS:
            excluded += 1
            continue
        try:
            row = phase_statistics(np.load(path, allow_pickle=False))
            rows.append({"file": path.name, **row})
        except (ValueError, OSError, TypeError) as error:
            errors.append({"file": path.name, "error": str(error)})
    if not rows:
        errors.append({"error": "没有可统计的有效样本"})
    metrics = ("frames", "raw_zero_fraction", "phase_std_radians", "real_outside_pi_fraction",
               "circular_temporal_rms", "channel_temporal_rms_median", "adjacent_phase_step_rms_radians")
    percentiles = {
        name: dict(zip(("p05", "p50", "p95"),
                       np.percentile([row[name] for row in rows], [5, 50, 95]).tolist()))
        for name in metrics
    } if rows else {}
    # 高斯角度扰动的预期单位圆位移 RMS；仅作尺度比较，不代表信噪比。
    noise_chord_rms = float(np.sqrt(-2 * np.expm1(-noise_std * (noise_std / 2))))
    return {
        "split": split, "valid_samples": len(rows), "excluded_samples": excluded,
        "input_kinds": dict(Counter(row["input_kind"] for row in rows)),
        "ambiguous_time_axis_samples": sum(row["ambiguous_time_axis"] for row in rows),
        "per_sample_percentiles": percentiles,
        "noise_std_radians": noise_std,
        "expected_noise_chord_rms": noise_chord_rms,
        "samples_temporal_rms_below_noise_chord": sum(row["circular_temporal_rms"] < noise_chord_rms for row in rows),
        "lowest_temporal_variation": sorted(rows, key=lambda row: row["circular_temporal_rms"])[:5],
        "error_count": len(errors), "errors_first_10": errors[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path("/data2/fanl/M2Voice/dataset/dt4_splitv2_clean"))
    parser.add_argument("--noise-std", type=float, default=0.02)
    args = parser.parse_args()
    if not np.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("noise-std 必须是非负有限数值")
    reports = []
    for split in ("train", "val"):
        try:
            reports.append(summarize_split(args.split_root, split, args.noise_std))
        except OSError as error:
            reports.append({"split": split, "error_count": 1, "error": str(error)})
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    return int(any(report["error_count"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
