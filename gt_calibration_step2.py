from pathlib import Path

gt_file = Path(
    "/data2/fanl/M2Voice/dataset/dt4/gt.txt"
)

txt_dir = Path(
    "/data2/fanl/M2Voice/dataset/dt4/txt_calibrated"
)


def normalize(text):
    return text.strip().lower().rstrip(".?!")


gt_set = {
    normalize(line)
    for line in gt_file.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
}

bad_files = []

for txt_file in sorted(txt_dir.glob("*.txt")):
    text = txt_file.read_text(
        encoding="utf-8"
    ).strip()

    if normalize(text) not in gt_set:
        bad_files.append((txt_file.name, text))

print(f"Total TXT files     : {len(list(txt_dir.glob('*.txt')))}")
print(f"Non-standard labels : {len(bad_files)}")

for filename, text in bad_files:
    print(f"{filename}: {text}")