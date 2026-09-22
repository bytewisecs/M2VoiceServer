import os
import shutil
import random
from pathlib import Path


ROOT_DIR = "/data2/fanl/M2Voice/dataset/dt2/"   # 改成你的原始数据目录
OUT_DIR = "/data2/fanl/M2Voice/dataset/split3/"  # 输出目录

SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3



def make_dirs(out_dir):
    for split in ["train", "val", "test"]:
        for sub in ["Audio", "mmLip", "mmVocal", "txt"]:
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)


def get_sample_id(audio_file):
    return audio_file.stem


def parse_paths(root_dir, sample_id):
    parts = sample_id.split("_")

    if len(parts) < 4:
        return None

    time_id = f"{parts[1]}_{parts[2]}"
    suffix = parts[3]

    return {
        "Audio": Path(root_dir) / "Audio" / f"{sample_id}.wav",
        "mmLip": Path(root_dir) / "mmLip" / f"mmW_{time_id}_Lip_{suffix}.npy",
        "mmVocal": Path(root_dir) / "mmVocal" / f"mmW_{time_id}_Vib_{suffix}.npy",
        "txt": Path(root_dir) / "txt" / f"{sample_id}.txt",
    }


def collect_samples(root_dir):
    audio_dir = Path(root_dir) / "Audio"

    samples = []

    for audio_file in sorted(audio_dir.glob("*.wav")):
        sample_id = get_sample_id(audio_file)
        paths = parse_paths(root_dir, sample_id)
        
        import pdb; pdb.set_trace()
        if paths is None:
            print(f"[Skip bad name] {audio_file.name}")
            continue

        missing = [k for k, v in paths.items() if not v.exists()]

        if missing:
            print(f"[Skip missing] {sample_id} missing {missing}")
            continue

        samples.append({
            "id": sample_id,
            "paths": paths
        })

    return samples


def copy_sample(sample, out_dir, split):
    sample_id = sample["id"]

    for sub, src_path in sample["paths"].items():
        dst_dir = Path(out_dir) / split / sub
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def main():
    random.seed(SEED)

    make_dirs(OUT_DIR)

    samples = collect_samples(ROOT_DIR)

    print(f"Total valid samples: {len(samples)}")

    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    split_map = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split, split_samples in split_map.items():
        print(f"{split}: {len(split_samples)}")

        for sample in split_samples:
            copy_sample(sample, OUT_DIR, split)

    print("Done.")


if __name__ == "__main__":
    main()
    
    
    

