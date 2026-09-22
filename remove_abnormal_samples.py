"""删除已确认排除的 5 个样本及其配套文件；默认只预览。"""
import argparse
from pathlib import Path


DEFAULT_ROOT = Path("/data2/fanl/M2Voice/dataset/dt4")
SAMPLE_IDS = (
    "audio_20260411_174447_s1",
    "audio_20260411_174614_s4",
    "audio_20260411_174756_s4",
    "audio_20260411_180543_s6",
    "audio_20260411_181349_s6",
)


def target_names():
    """使用完整文件名白名单，避免 s1 误匹配 s10 或误删整个会话。"""
    names = {folder: set() for folder in ("Audio", "txt", "txt_calibrated", "mmLip", "mmVocal")}
    for sample_id in SAMPLE_IDS:
        group_id, suffix = sample_id.removeprefix("audio_").rsplit("_", 1)
        names["Audio"].update(sample_id + extension for extension in (".wav", ".mp3", ".flac"))
        for folder in ("txt", "txt_calibrated"):
            names[folder].update((sample_id + ".txt", sample_id + ".txt.done.json"))
        names["mmLip"].add(f"mmW_{group_id}_Lip_{suffix}.npy")
        names["mmVocal"].add(f"mmW_{group_id}_Vib_{suffix}.npy")
    return names


def collect_targets(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    # 先完整检查所有路径，再开始删除，避免路径配置错误时删到一半。
    for folder in ("Audio", "txt_calibrated"):
        if not (root / folder).is_dir():
            raise FileNotFoundError(f"数据目录不存在：{root / folder}")

    targets = []
    for folder, names in target_names().items():
        directory = root / folder
        if directory.is_symlink():
            raise ValueError(f"不处理符号链接目录：{directory}")
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        allowed = {name.lower() for name in names}
        for path in sorted(directory.iterdir()):
            if path.name.lower() not in allowed:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"目标不是普通文件：{path}")
            targets.append(path)
    return targets


def remove_samples(root, apply=False):
    targets = collect_targets(root)
    print(f"Mode: {'DELETE' if apply else 'PREVIEW'}")
    print(f"Matched files: {len(targets)}")
    for path in targets:
        if apply:
            path.unlink()
        print(f"{'Deleted' if apply else 'Would delete'}: {path}")
    if not apply:
        print("预览完成，未删除文件。使用 --apply 执行删除。")
    return len(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="实际删除白名单中的样本文件")
    args = parser.parse_args()
    try:
        remove_samples(args.root, apply=args.apply)
    except (OSError, ValueError) as exc:
        print(f"Cleanup failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
