import argparse
import re
from pathlib import Path


ROOT_DIR = Path("/data2/fanl/M2Voice/dataset/dt4")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def normalize(text):
    return text.strip().lower().rstrip(".?!").strip()


def validate_labels(gt_file, txt_dir, audio_dir):
    """以音频为基准，检查标签完整性及每个文件允许的两个 GT。"""
    gt_file, txt_dir, audio_dir = map(Path, (gt_file, txt_dir, audio_dir))
    for directory in (txt_dir, audio_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"目录不存在：{directory}")

    gt_sentences = [
        normalize(line)
        for line in gt_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(gt_sentences) != 20 or not all(gt_sentences):
        raise ValueError("GT 文件必须包含 20 条非空标准句。")
    if len(set(gt_sentences)) != 20:
        raise ValueError("GT 标准化后存在重复句子。")

    audio_files = sorted(
        path for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    txt_files = sorted(
        path for path in txt_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )
    errors = []
    if not audio_files:
        errors.append(f"音频目录为空或没有支持的音频文件：{audio_dir}")
    if not txt_files:
        errors.append(f"没有标签文件：{txt_dir}")

    audio_stems = set()
    for path in audio_files:
        if path.stem in audio_stems:
            errors.append(f"音频同名冲突：{path.stem}")
        audio_stems.add(path.stem)

    txt_stems = set()
    for path in txt_files:
        if path.stem in txt_stems:
            errors.append(f"标签同名冲突：{path.stem}")
        txt_stems.add(path.stem)
        if path.stem not in audio_stems:
            errors.append(f"标签没有对应音频：{path.name}")

        match = re.search(r"_s(10|[1-9])\.txt$", path.name, re.IGNORECASE)
        if match is None:
            errors.append(f"标签文件名后缀无效：{path.name}")
            continue
        try:
            text = normalize(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as exc:
            errors.append(f"标签读取失败：{path.name}: {exc}")
            continue

        suffix_id = int(match.group(1))
        allowed = (gt_sentences[suffix_id - 1], gt_sentences[suffix_id + 9])
        if text not in allowed:
            errors.append(
                f"标签不匹配：{path.name} 只允许 GT {suffix_id} 或 "
                f"GT {suffix_id + 10}，实际内容：{text!r}"
            )

    for stem in sorted(audio_stems - txt_stems):
        errors.append(f"缺少标签：{stem}.txt")

    print(f"Total audio files   : {len(audio_files)}")
    print(f"Total TXT files     : {len(txt_files)}")
    print(f"Validation errors   : {len(errors)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="校验校准标签的完整性和 GT 候选范围")
    parser.add_argument("--gt-file", type=Path, default=ROOT_DIR / "gt.txt")
    parser.add_argument("--txt-dir", type=Path, default=ROOT_DIR / "txt_calibrated")
    parser.add_argument("--audio-dir", type=Path, default=ROOT_DIR / "Audio")
    args = parser.parse_args()
    try:
        errors = validate_labels(args.gt_file, args.txt_dir, args.audio_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        return 1
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
