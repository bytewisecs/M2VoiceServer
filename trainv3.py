import os
import re
import math
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 64
    config.mlp_dim = 256
    config.num_heads = 4
    config.num_layers = 3
    config.dropout = 0.2
    return config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ConvFrontend1D(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),

            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),

            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class LipVocalTextClassifier(nn.Module):
    def __init__(self, config, num_classes, target_seq_len=128):
        super().__init__()

        hidden = config.hidden_size
        self.target_seq_len = target_seq_len

        self.lip_frontend = ConvFrontend1D(
            in_channels=1,
            hidden_size=hidden
        )

        self.vocal_frontend = ConvFrontend1D(
            in_channels=256,
            hidden_size=hidden
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.pos_encoder = PositionalEncoding(hidden, max_len=target_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, lip, vocal):
        lip_feat = self.lip_frontend(lip)
        vocal_feat = self.vocal_frontend(vocal)

        lip_feat = F.interpolate(
            lip_feat,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False
        )

        vocal_feat = F.interpolate(
            vocal_feat,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False
        )

        lip_feat = lip_feat.transpose(1, 2)
        vocal_feat = vocal_feat.transpose(1, 2)

        x = torch.cat([lip_feat, vocal_feat], dim=-1)
        x = self.fusion(x)
        x = self.pos_encoder(x)

        x = self.encoder(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


class M2VoiceTextDataset(Dataset):
    def __init__(self, root_dir, label_texts, split_name):
        self.root_dir = root_dir
        self.split_name = split_name

        self.audio_dir = os.path.join(root_dir, "Audio")
        self.lip_dir = os.path.join(root_dir, "mmLip")
        self.vocal_dir = os.path.join(root_dir, "mmVocal")
        self.txt_dir = os.path.join(root_dir, "txt")

        self.samples = []
        # train/val/test共用由gt.txt确定的同一套20类映射。
        self.label_texts = list(label_texts)
        self.text_to_label = {
            text: idx for idx, text in enumerate(self.label_texts)
        }

        self._parse_dataset()

        print(
            f"Loaded {self.split_name} samples: {len(self.samples)} "
            f"from {self.root_dir}"
        )
        self.print_class_distribution()

    def _parse_dataset(self):
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(self.audio_dir)

        if not os.path.exists(self.lip_dir):
            raise FileNotFoundError(self.lip_dir)

        if not os.path.exists(self.vocal_dir):
            raise FileNotFoundError(self.vocal_dir)

        if not os.path.exists(self.txt_dir):
            raise FileNotFoundError(self.txt_dir)

        temp_samples = []

        for audio_filename in sorted(os.listdir(self.audio_dir)):
            if not audio_filename.endswith(".wav"):
                continue

            name = audio_filename.replace(".wav", "")
            parts = name.split("_")

            if len(parts) < 4:
                continue

            time_id = f"{parts[1]}_{parts[2]}"
            suffix = parts[3]

            lip_path = os.path.join(
                self.lip_dir,
                f"mmW_{time_id}_Lip_{suffix}.npy"
            )

            vocal_path = os.path.join(
                self.vocal_dir,
                f"mmW_{time_id}_Vib_{suffix}.npy"
            )

            txt_path = os.path.join(
                self.txt_dir,
                f"{name}.txt"
            )

            if not os.path.exists(lip_path):
                print(f"[Missing Lip] {lip_path}")
                continue

            if not os.path.exists(vocal_path):
                print(f"[Missing Vocal] {vocal_path}")
                continue

            if not os.path.exists(txt_path):
                print(f"[Missing Txt] {txt_path}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                text = clean_text(f.read())

            if len(text) == 0:
                print(f"[Empty Txt] {txt_path}")
                continue

            temp_samples.append({
                "id": name,
                "lip_path": lip_path,
                "vocal_path": vocal_path,
                "txt_path": txt_path,
                "text": text,
            })

        for sample in temp_samples:
            if sample["text"] not in self.text_to_label:
                raise ValueError(
                    f"[{self.split_name}] 检测到不属于20条GT的标签：\n"
                    f"文件：{sample['txt_path']}\n"
                    f"文本：{sample['text']}"
                )

            sample["label"] = self.text_to_label[sample["text"]]
            self.samples.append(sample)

        if len(self.samples) == 0:
            raise ValueError(
                f"{self.split_name}数据集为空：{self.root_dir}"
            )

    def print_class_distribution(self):
        counts = Counter(sample["label"] for sample in self.samples)
        missing = []

        print(f"\n========== {self.split_name} Class Distribution ==========")
        for label_id, text in enumerate(self.label_texts):
            count = counts.get(label_id, 0)
            print(f"{label_id:02d}: {count:3d} | {text}")
            if count == 0:
                missing.append(label_id)

        if missing:
            print(
                f"[Warning] {self.split_name}缺少类别：{missing}。"
                "标签ID仍与其他数据集保持一致。"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        lip = np.load(sample["lip_path"])
        vocal = np.load(sample["vocal_path"])

        lip = np.asarray(lip, dtype=np.float32)

        if np.iscomplexobj(vocal):
            vocal = np.abs(vocal)

        vocal = np.asarray(vocal, dtype=np.float32)

        lip = torch.from_numpy(lip).float()
        vocal = torch.from_numpy(vocal).float()

        lip = (lip - lip.mean()) / (lip.std() + 1e-6)
        vocal = (vocal - vocal.mean()) / (vocal.std() + 1e-6)

        return {
            "id": sample["id"],
            "lip": lip,
            "vocal": vocal,
            "label": sample["label"],
            "text": sample["text"],
        }


def collate_fn(batch):
    ids = [b["id"] for b in batch]
    texts = [b["text"] for b in batch]

    lips = pad_sequence(
        [b["lip"] for b in batch],
        batch_first=True
    ).unsqueeze(1)

    vocals = pad_sequence(
        [b["vocal"] for b in batch],
        batch_first=True
    )

    labels = torch.tensor(
        [b["label"] for b in batch],
        dtype=torch.long
    )

    return {
        "id": ids,
        "lip": lips,
        "vocal": vocals,
        "label": labels,
        "text": texts,
    }


def evaluate(model, loader, device, num_classes=None):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    class_correct = None
    class_total = None
    if num_classes is not None:
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            lip = batch["lip"].to(device)
            vocal = batch["vocal"].to(device)
            labels = batch["label"].to(device)

            logits = model(lip, vocal)
            loss = criterion(logits, labels)

            pred = logits.argmax(dim=-1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            if num_classes is not None:
                for label, prediction in zip(
                    labels.cpu().tolist(),
                    pred.cpu().tolist()
                ):
                    class_total[label] += 1
                    if label == prediction:
                        class_correct[label] += 1

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)

    return avg_loss, acc, class_correct, class_total


def load_label_texts(gt_file):
    if not os.path.isfile(gt_file):
        raise FileNotFoundError(f"GT文件不存在：{gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        label_texts = [
            clean_text(line)
            for line in f
            if clean_text(line)
        ]

    if len(label_texts) != 20:
        raise ValueError(
            f"GT文件应包含20条非空句子，当前为{len(label_texts)}条："
            f"{gt_file}"
        )

    if len(set(label_texts)) != 20:
        raise ValueError("GT文件清洗后存在重复句子。")

    return label_texts


def validate_split_overlap(train_dataset, val_dataset, test_dataset):
    split_ids = {
        "train": {sample["id"] for sample in train_dataset.samples},
        "val": {sample["id"] for sample in val_dataset.samples},
        "test": {sample["id"] for sample in test_dataset.samples},
    }

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for left, right in pairs:
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            examples = sorted(overlap)[:10]
            raise ValueError(
                f"{left}与{right}存在{len(overlap)}个重复样本，"
                f"例如：{examples}"
            )

    print("\nSplit overlap check: passed")


def train():
    split_root = "/data2/fanl/M2Voice/dataset/dt3_splitv2"
    train_path = os.path.join(split_root, "train")
    val_path = os.path.join(split_root, "val")
    test_path = os.path.join(split_root, "test")
    gt_file = "/data2/fanl/M2Voice/dataset/dt3/gt.txt"

    checkpoint_dir = "runs/checkpoints_text_cls_split2_v3"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    seed = 42
    num_epochs = 100
    batch_size = 8
    lr = 1e-4
    target_seq_len = 128
    early_stopping_patience = 20

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 标签顺序完全由gt.txt决定，三套数据共用同一个text_to_label。
    label_texts = load_label_texts(gt_file)

    print("\n========== Shared Label Mapping ==========")
    for label_id, text in enumerate(label_texts):
        print(f"{label_id:02d}: {text}")

    train_dataset = M2VoiceTextDataset(
        train_path,
        label_texts,
        split_name="train"
    )
    val_dataset = M2VoiceTextDataset(
        val_path,
        label_texts,
        split_name="val"
    )
    test_dataset = M2VoiceTextDataset(
        test_path,
        label_texts,
        split_name="test"
    )

    validate_split_overlap(
        train_dataset,
        val_dataset,
        test_dataset
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(checkpoint_dir, "label_texts.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(label_texts, f, ensure_ascii=False, indent=2)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    config = get_config()

    model = LipVocalTextClassifier(
        config=config,
        num_classes=len(label_texts),
        target_seq_len=target_seq_len
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\n========== Start Training ==========")

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            lip = batch["lip"].to(device)
            vocal = batch["vocal"].to(device)
            labels = batch["label"].to(device)

            logits = model(lip, vocal)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                gt_id = labels[0].item()
                pred_id = pred[0].item()

                print(
                    f"\nEpoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

                print(f"ID      : {batch['id'][0]}")
                print(f"Target  : {label_texts[gt_id]}")
                print(f"Predict : {label_texts[pred_id]}")
                print(f"Train Acc Running: {correct / max(1, total) * 100:.2f}%")

        scheduler.step()

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            device
        )

        print("\n========== Epoch Summary ==========")
        print(f"Epoch      : {epoch + 1}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc * 100:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc * 100:.2f}%")

        improved = (
            val_acc > best_val_acc
            or (
                abs(val_acc - best_val_acc) < 1e-12
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "label_texts": label_texts,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "hidden_size": config.hidden_size,
                        "mlp_dim": config.mlp_dim,
                        "num_heads": config.num_heads,
                        "num_layers": config.num_layers,
                        "dropout": config.dropout,
                        "target_seq_len": target_seq_len,
                    },
                    "split_root": split_root,
                },
                checkpoint_path
            )

            print(
                f"Saved best model. Val Acc = {best_val_acc * 100:.2f}%, "
                f"Val Loss = {best_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{early_stopping_patience}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best Val Acc = {best_val_acc * 100:.2f}%"
            )
            break

    print("\n========== Load Best Model ==========")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint["label_texts"] != label_texts:
        raise ValueError("Checkpoint标签映射与当前GT标签映射不一致。")

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best epoch    : {checkpoint['epoch']}")
    print(f"Best Val Loss : {checkpoint['best_val_loss']:.4f}")
    print(f"Best Val Acc  : {checkpoint['best_val_acc'] * 100:.2f}%")

    # 测试集只在最佳模型确定后评估一次。
    test_loss, test_acc, class_correct, class_total = evaluate(
        model,
        test_loader,
        device,
        num_classes=len(label_texts)
    )

    print("\n========== Test Summary ==========")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc * 100:.2f}%")

    per_class_results = []

    print("\n========== Per-Class Test Accuracy ==========")
    for label_id, text in enumerate(label_texts):
        total_count = class_total[label_id]
        correct_count = class_correct[label_id]
        class_acc = (
            correct_count / total_count
            if total_count > 0
            else None
        )

        if class_acc is None:
            acc_text = "N/A"
        else:
            acc_text = f"{class_acc * 100:.2f}%"

        print(
            f"{label_id:02d}: {correct_count}/{total_count} "
            f"({acc_text}) | {text}"
        )

        per_class_results.append({
            "label_id": label_id,
            "text": text,
            "correct": correct_count,
            "total": total_count,
            "accuracy": class_acc,
        })

    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "best_val_accuracy": checkpoint["best_val_acc"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "per_class_test_results": per_class_results,
    }

    results_path = os.path.join(
        checkpoint_dir,
        "test_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCheckpoint   : {checkpoint_path}")
    print(f"Test results : {results_path}")


if __name__ == "__main__":
    train()
    
