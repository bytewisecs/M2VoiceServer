import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from remove_abnormal_samples import SAMPLE_IDS as EXCLUDED_SAMPLE_IDS
from mDataloader.exclusions import INVALID_GROUPS


# ============================================================
# 1. 路径与划分配置
# ============================================================
ROOT_DIR = Path("/data2/fanl/M2Voice/dataset/dt4")
OUT_DIR = Path("/data2/fanl/M2Voice/dataset/dt4_splitv2_clean")

AUDIO_DIR = ROOT_DIR / "Audio"
LIP_DIR = ROOT_DIR / "mmLip"
VOCAL_DIR = ROOT_DIR / "mmVocal"
TXT_DIR = ROOT_DIR / "txt_calibrated"
GT_FILE = ROOT_DIR / "gt.txt"

SEED = 42
SPLIT_RATIOS = {
    "train": 0.6,
    "val": 0.3,
    "test": 0.1,
}

# 仅允许已明确删除的 5 个样本缺席；其他会话仍必须完整。
EXPECTED_SUFFIXES = {f"s{i}" for i in range(1, 11)}

# audio_20260411_165321_s1.wav
AUDIO_NAME_PATTERN = re.compile(
    r"^audio_(?P<group_id>\d{8}_\d{6})_(?P<suffix>s(?:10|[1-9]))$",
    re.IGNORECASE,
)


# ============================================================
# 2. 文本和GT处理
# ============================================================
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_gt_labels(gt_file):
    if not gt_file.is_file():
        raise FileNotFoundError(f"GT文件不存在：{gt_file}")

    with gt_file.open("r", encoding="utf-8") as f:
        labels = [clean_text(line) for line in f if clean_text(line)]

    if len(labels) != 20:
        raise ValueError(
            f"GT文件应包含20条非空句子，当前为{len(labels)}条：{gt_file}"
        )

    if len(set(labels)) != 20:
        raise ValueError("GT文件清洗后存在重复句子。")

    return labels


# ============================================================
# 3. 样本收集与完整性检查
# ============================================================
def build_sample_paths(audio_file, group_id, suffix):
    sample_id = audio_file.stem

    return {
        "Audio": audio_file,
        "mmLip": LIP_DIR / f"mmW_{group_id}_Lip_{suffix}.npy",
        "mmVocal": VOCAL_DIR / f"mmW_{group_id}_Vib_{suffix}.npy",
        "txt": TXT_DIR / f"{sample_id}.txt",
    }


def collect_samples(label_texts):
    required_dirs = [AUDIO_DIR, LIP_DIR, VOCAL_DIR, TXT_DIR]
    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{directory}")

    text_to_label = {
        text: label_id for label_id, text in enumerate(label_texts)
    }

    samples = []
    errors = []

    audio_files = sorted(path for path in AUDIO_DIR.glob("*.wav") if path.is_file())
    if not audio_files:
        raise ValueError(f"没有找到WAV文件：{AUDIO_DIR}")

    for audio_file in audio_files:
        if audio_file.stem in EXCLUDED_SAMPLE_IDS:
            errors.append(
                f"待排除样本仍存在：{audio_file.name}，请先执行 remove_abnormal_samples.py --apply。"
            )
            continue
        match = AUDIO_NAME_PATTERN.fullmatch(audio_file.stem)

        if not match:
            errors.append(f"文件名格式错误：{audio_file.name}")
            continue

        group_id = match.group("group_id")
        if group_id in INVALID_GROUPS:
            print(f"[Skip known invalid group] {audio_file.name}")
            continue
        suffix = match.group("suffix").lower()
        suffix_number = int(suffix[1:])
        paths = build_sample_paths(audio_file, group_id, suffix)

        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            details = ", ".join(
                f"{name}={paths[name]}" for name in missing
            )
            errors.append(f"{audio_file.stem} 缺少文件：{details}")
            continue

        with paths["txt"].open("r", encoding="utf-8") as f:
            label_text = clean_text(f.read())

        if label_text not in text_to_label:
            errors.append(
                f"{paths['txt']} 的标签不属于20条GT：{label_text!r}"
            )
            continue

        label_id = text_to_label[label_text]

        # GT 1和11均对应s1，GT 2和12均对应s2，以此类推。
        expected_suffix_number = label_id % 10 + 1
        if suffix_number != expected_suffix_number:
            errors.append(
                f"{audio_file.name} 的后缀为s{suffix_number},"
                f"但标签ID {label_id + 1} 应对应s{expected_suffix_number},"
                f"{label_text}"
            )
            continue

        command_set = 0 if label_id < 10 else 1

        samples.append({
            "sample_id": audio_file.stem,
            "group_id": group_id,
            "suffix": suffix,
            "command_set": command_set,
            "label_id": label_id,
            "label_text": label_text,
            "paths": paths,
        })

    if errors:
        preview = "\n".join(errors[:50])
        remaining = len(errors) - min(50, len(errors))
        if remaining > 0:
            preview += f"\n……另外还有{remaining}个错误。"

        raise ValueError(
            f"数据完整性检查失败，共发现{len(errors)}个错误：\n{preview}"
        )

    if not samples:
        raise ValueError("排除已知无效会话后，没有可用于划分的样本。")
    return samples


def validate_groups(samples):
    excluded_by_group = defaultdict(set)
    for sample_id in EXCLUDED_SAMPLE_IDS:
        match = AUDIO_NAME_PATTERN.fullmatch(sample_id)
        if match is None:
            raise ValueError(f"无效的排除样本编号：{sample_id}")
        excluded_by_group[match.group("group_id")].add(match.group("suffix").lower())

    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_id"]].append(sample)

    errors = []

    for group_id, group_samples in sorted(groups.items()):
        suffixes = {sample["suffix"] for sample in group_samples}
        command_sets = {sample["command_set"] for sample in group_samples}
        label_ids = [sample["label_id"] for sample in group_samples]

        expected = EXPECTED_SUFFIXES - excluded_by_group[group_id]
        if len(group_samples) != len(expected):
            errors.append(
                f"会话{group_id}应包含{len(expected)}个样本，当前为{len(group_samples)}个。"
            )

        if suffixes != expected:
            errors.append(
                f"会话{group_id}的后缀不符合预期："
                f"缺失={sorted(expected - suffixes)}，多余={sorted(suffixes - expected)}"
            )

        if len(command_sets) != 1:
            errors.append(
                f"会话{group_id}同时包含前10类和后10类标签。"
            )

        if len(set(label_ids)) != len(label_ids):
            errors.append(f"会话{group_id}存在重复标签。")

    if errors:
        raise ValueError("会话完整性检查失败：\n" + "\n".join(errors))

    return groups


# ============================================================
# 4. 按会话和指令组进行6:3:1划分（比例按会话数计算）
# ============================================================
def allocate_counts(total, ratios):
    """使用最大余数法分配数量，确保三部分之和严格等于total。"""
    names = list(ratios.keys())
    raw_counts = [total * ratios[name] for name in names]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)

    remainder_order = sorted(
        range(len(names)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )

    for index in remainder_order[:remaining]:
        counts[index] += 1

    return dict(zip(names, counts))


def split_groups(groups):
    if set(SPLIT_RATIOS) != {"train", "val", "test"} or any(
        not 0 < ratio < 1 for ratio in SPLIT_RATIOS.values()
    ):
        raise ValueError("train、val、test 的划分比例必须介于 0 和 1 之间。")
    ratio_sum = sum(SPLIT_RATIOS.values())
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"划分比例之和必须为1，当前为{ratio_sum}")

    group_ids_by_set = defaultdict(list)

    for group_id, group_samples in groups.items():
        command_set = group_samples[0]["command_set"]
        group_ids_by_set[command_set].append(group_id)

    split_to_group_ids = {name: [] for name in SPLIT_RATIOS}

    for command_set in [0, 1]:
        group_ids = sorted(group_ids_by_set[command_set])

        if not group_ids:
            raise ValueError(f"没有找到command_set={command_set}的采集会话。")

        rng = random.Random(SEED + command_set)
        rng.shuffle(group_ids)

        counts = allocate_counts(len(group_ids), SPLIT_RATIOS)
        start = 0

        for split_name in SPLIT_RATIOS:
            end = start + counts[split_name]
            split_to_group_ids[split_name].extend(group_ids[start:end])
            start = end

        print(
            f"Command set {command_set + 1}: {len(group_ids)} groups -> "
            + ", ".join(
                f"{name}={counts[name]}" for name in SPLIT_RATIOS
            )
        )

    # 最终检查：同一会话不能出现在多个集合。
    split_sets = {
        name: set(group_ids)
        for name, group_ids in split_to_group_ids.items()
    }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise RuntimeError(
                f"{left}与{right}存在重复会话：{sorted(overlap)}"
            )

    return split_to_group_ids


def validate_split_plan(groups, split_to_group_ids, label_count):
    """复制前验证会话唯一归属、无遗漏及每个集合的类别覆盖。"""
    assigned = [group_id for ids in split_to_group_ids.values() for group_id in ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError("划分计划存在重复会话。")
    if set(assigned) != set(groups):
        raise ValueError("划分计划遗漏会话或包含未知会话。")

    for split_name in SPLIT_RATIOS:
        ids = split_to_group_ids[split_name]
        rows = [sample for group_id in ids for sample in groups[group_id]]
        counts = Counter(sample["label_id"] for sample in rows)
        missing = [label + 1 for label in range(label_count) if counts[label] == 0]
        if missing:
            raise ValueError(f"{split_name}划分计划缺少 GT 类别：{missing}，未复制文件。")
        print(f"Plan {split_name:5s}: groups={len(ids)}, samples={len(rows)}")


# ============================================================
# 5. 输出目录、复制及清单
# ============================================================
def prepare_output_dirs():
    source = ROOT_DIR.resolve()
    output = OUT_DIR.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输入和输出目录不能相同，也不能互相包含。")
    if OUT_DIR.is_symlink():
        raise ValueError(f"输出目录不能是符号链接：{OUT_DIR}")
    if OUT_DIR.exists():
        if not OUT_DIR.is_dir():
            raise NotADirectoryError(OUT_DIR)
        if any(OUT_DIR.iterdir()):
            raise FileExistsError(
                f"输出目录非空：{OUT_DIR}\n"
                "为避免旧划分残留，请通过 --out-dir 指定新的空目录。"
            )

    for split_name in SPLIT_RATIOS:
        for subdir in ["Audio", "mmLip", "mmVocal", "txt"]:
            (OUT_DIR / split_name / subdir).mkdir(parents=True, exist_ok=True)


def copy_splits(groups, split_to_group_ids):
    manifest_rows = []

    for split_name, group_ids in split_to_group_ids.items():
        for group_id in sorted(group_ids):
            for sample in sorted(
                groups[group_id],
                key=lambda item: item["suffix"],
            ):
                for subdir, source_path in sample["paths"].items():
                    destination_path = (
                        OUT_DIR / split_name / subdir / source_path.name
                    )
                    shutil.copy2(source_path, destination_path)

                manifest_rows.append({
                    "split": split_name,
                    "group_id": group_id,
                    "sample_id": sample["sample_id"],
                    "suffix": sample["suffix"],
                    "command_set": sample["command_set"] + 1,
                    "label_id": sample["label_id"],
                    "gt_number": sample["label_id"] + 1,
                    "label_text": sample["label_text"],
                })

    manifest_path = OUT_DIR / "split_manifest.csv"
    fieldnames = [
        "split",
        "group_id",
        "sample_id",
        "suffix",
        "command_set",
        "label_id",
        "gt_number",
        "label_text",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_rows


# ============================================================
# 6. 输出结果验证与统计
# ============================================================
def verify_and_summarize(manifest_rows, label_texts, split_to_group_ids):
    summary = {
        "seed": SEED,
        "ratios": SPLIT_RATIOS,
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUT_DIR),
        "ratio_unit": "acquisition_group",
        "excluded_sample_ids": list(EXCLUDED_SAMPLE_IDS),
        "excluded_group_ids": sorted(INVALID_GROUPS),
        "sample_count": len(manifest_rows),
        "group_count": sum(len(ids) for ids in split_to_group_ids.values()),
        "splits": {},
    }

    print("\n========== Split Summary ==========")

    for split_name in SPLIT_RATIOS:
        rows = [row for row in manifest_rows if row["split"] == split_name]
        class_counts = Counter(row["label_id"] for row in rows)

        modality_counts = {}
        expected_extensions = {
            "Audio": ".wav",
            "mmLip": ".npy",
            "mmVocal": ".npy",
            "txt": ".txt",
        }

        for subdir, extension in expected_extensions.items():
            modality_counts[subdir] = len(
                list((OUT_DIR / split_name / subdir).glob(f"*{extension}"))
            )

        if len(set(modality_counts.values())) != 1:
            raise RuntimeError(
                f"{split_name}四种模态数量不一致：{modality_counts}"
            )

        if modality_counts["Audio"] != len(rows):
            raise RuntimeError(
                f"{split_name}清单数量与输出文件数量不一致。"
            )

        missing_classes = [
            label_id
            for label_id in range(len(label_texts))
            if class_counts.get(label_id, 0) == 0
        ]

        if missing_classes:
            raise RuntimeError(
                f"{split_name}缺少类别：{missing_classes}"
            )

        print(
            f"{split_name:5s}: samples={len(rows):3d}, "
            f"groups={len(split_to_group_ids[split_name]):2d}, "
            f"modalities={modality_counts}"
        )

        print(
            "       class counts: "
            + ", ".join(
                f"{label_id:02d}:{class_counts[label_id]}"
                for label_id in range(len(label_texts))
            )
        )

        summary["splits"][split_name] = {
            "sample_count": len(rows),
            "group_count": len(split_to_group_ids[split_name]),
            "group_ids": sorted(split_to_group_ids[split_name]),
            "modality_counts": modality_counts,
            "class_counts": {
                str(label_id): class_counts[label_id]
                for label_id in range(len(label_texts))
            },
        }

    summary_path = OUT_DIR / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_path


# ============================================================
# 7. 主程序
# ============================================================
def main():
    global ROOT_DIR, OUT_DIR, AUDIO_DIR, LIP_DIR, VOCAL_DIR, TXT_DIR, GT_FILE
    parser = argparse.ArgumentParser(description="排除指定异常样本后，按采集会话分层划分数据集")
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--check-only", action="store_true", help="仅校验输入与划分计划，不复制文件")
    args = parser.parse_args()
    ROOT_DIR, OUT_DIR = args.root, args.out_dir
    AUDIO_DIR, LIP_DIR = ROOT_DIR / "Audio", ROOT_DIR / "mmLip"
    VOCAL_DIR, TXT_DIR = ROOT_DIR / "mmVocal", ROOT_DIR / "txt_calibrated"
    GT_FILE = ROOT_DIR / "gt.txt"

    print(f"Source directory : {ROOT_DIR}")
    print(f"Output directory : {OUT_DIR}")
    print(f"Calibrated TXT   : {TXT_DIR}")
    print(f"Random seed      : {SEED}\n")

    label_texts = load_gt_labels(GT_FILE)
    samples = collect_samples(label_texts)
    groups = validate_groups(samples)

    print(f"Valid samples : {len(samples)}")
    print(f"Valid groups  : {len(groups)}\n")

    split_to_group_ids = split_groups(groups)
    validate_split_plan(groups, split_to_group_ids, len(label_texts))
    if args.check_only:
        print("\n检查通过，未创建输出目录或复制文件。")
        return

    prepare_output_dirs()
    manifest_rows = copy_splits(groups, split_to_group_ids)
    summary_path = verify_and_summarize(
        manifest_rows,
        label_texts,
        split_to_group_ids,
    )

    print("\nDone.")
    print(f"Dataset  : {OUT_DIR}")
    print(f"Manifest : {OUT_DIR / 'split_manifest.csv'}")
    print(f"Summary  : {summary_path}")


if __name__ == "__main__":
    main()
