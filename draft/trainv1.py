import os
import re
import math
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.tensorboard import SummaryWriter


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 256
    config.mlp_dim = 512
    config.num_heads = 4
    config.num_layers = 4
    config.dropout = 0.3
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
            nn.Conv1d(in_channels, hidden_size // 2, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),

            nn.Conv1d(hidden_size // 2, hidden_size, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),

            nn.Conv1d(hidden_size, hidden_size, 3, stride=1, padding=1),
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

        self.lip_frontend = ConvFrontend1D(1, hidden)
        self.vocal_frontend = ConvFrontend1D(256, hidden)

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

        return self.classifier(x)


def build_label_mapping(train_root):
    txt_dir = os.path.join(train_root, "txt")
    texts = []

    for fn in sorted(os.listdir(txt_dir)):
        if not fn.endswith(".txt"):
            continue

        with open(os.path.join(txt_dir, fn), "r", encoding="utf-8") as f:
            text = clean_text(f.read())

        if len(text) > 0:
            texts.append(text)

    label_texts = sorted(list(set(texts)))
    text_to_label = {t: i for i, t in enumerate(label_texts)}

    print("\n========== Label Mapping from Train Set ==========")
    for i, text in enumerate(label_texts):
        print(f"{i:03d}: {text}")

    print(f"\nTotal classes: {len(label_texts)}")

    os.makedirs("checkpoints_text_cls", exist_ok=True)
    with open("checkpoints_text_cls/label_texts.json", "w", encoding="utf-8") as f:
        json.dump(label_texts, f, ensure_ascii=False, indent=2)

    return label_texts, text_to_label


class M2VoiceTextDataset(Dataset):
    def __init__(self, root_dir, text_to_label, split_name="train"):
        self.root_dir = root_dir
        self.split_name = split_name

        self.audio_dir = os.path.join(root_dir, "Audio")
        self.lip_dir = os.path.join(root_dir, "mmLip")
        self.vocal_dir = os.path.join(root_dir, "mmVocal")
        self.txt_dir = os.path.join(root_dir, "txt")

        self.text_to_label = text_to_label
        self.samples = []

        self._parse_dataset()

        print(f"[{split_name}] Loaded samples: {len(self.samples)}")

        counter = Counter([s["label"] for s in self.samples])
        print(f"[{split_name}] Class distribution:")
        for label, count in sorted(counter.items()):
            print(f"  class {label:03d}: {count}")

    def _parse_dataset(self):
        for d in [self.audio_dir, self.lip_dir, self.vocal_dir, self.txt_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(d)

        for audio_filename in sorted(os.listdir(self.audio_dir)):
            if not audio_filename.endswith(".wav"):
                continue

            name = audio_filename.replace(".wav", "")
            parts = name.split("_")

            if len(parts) < 4:
                print(f"[Bad name] {audio_filename}")
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
                print(f"[{self.split_name}] Missing Lip: {lip_path}")
                continue

            if not os.path.exists(vocal_path):
                print(f"[{self.split_name}] Missing Vocal: {vocal_path}")
                continue

            if not os.path.exists(txt_path):
                print(f"[{self.split_name}] Missing Txt: {txt_path}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                text = clean_text(f.read())

            if len(text) == 0:
                print(f"[{self.split_name}] Empty Txt: {txt_path}")
                continue

            if text not in self.text_to_label:
                print(f"[{self.split_name}] Unknown label text, skipped: {text}")
                continue

            self.samples.append({
                "id": name,
                "lip_path": lip_path,
                "vocal_path": vocal_path,
                "txt_path": txt_path,
                "text": text,
                "label": self.text_to_label[text],
            })

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
        
        if self.split_name == "train":
            lip = lip + torch.randn_like(lip) * 0.02
            vocal = vocal + torch.randn_like(vocal) * 0.02

            scale = torch.empty(1).uniform_(0.9, 1.1)
            lip = lip * scale
            vocal = vocal * scale

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


def evaluate(model, loader, label_texts, device, name="val"):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    wrong_cases = []

    with torch.no_grad():
        for batch in loader:
            lip = batch["lip"].to(device)
            vocal = batch["vocal"].to(device)
            labels = batch["label"].to(device)

            logits = model(lip, vocal)
            loss = criterion(logits, labels)

            pred = logits.argmax(dim=-1)

            total_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                if pred[i].item() != labels[i].item():
                    wrong_cases.append({
                        "id": batch["id"][i],
                        "target": label_texts[labels[i].item()],
                        "predict": label_texts[pred[i].item()],
                    })

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)

    print(f"\n[{name}] Loss: {avg_loss:.4f} | Acc: {acc * 100:.2f}%")

    if len(wrong_cases) > 0:
        print(f"[{name}] Wrong examples:")
        for item in wrong_cases[:10]:
            print(f"  ID: {item['id']}")
            print(f"  GT: {item['target']}")
            print(f"  PR: {item['predict']}")

    return avg_loss, acc


def train():
    DATA_ROOT = "/data2/fanl/M2Voice/dataset/split2"

    TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
    VAL_ROOT = os.path.join(DATA_ROOT, "val")
    TEST_ROOT = os.path.join(DATA_ROOT, "test")

    num_epochs = 30
    batch_size = 8
    lr = 1e-4
    target_seq_len = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_texts, text_to_label = build_label_mapping(TRAIN_ROOT)

    train_dataset = M2VoiceTextDataset(
        TRAIN_ROOT,
        text_to_label=text_to_label,
        split_name="train"
    )

    val_dataset = M2VoiceTextDataset(
        VAL_ROOT,
        text_to_label=text_to_label,
        split_name="val"
    )

    test_dataset = M2VoiceTextDataset(
        TEST_ROOT,
        text_to_label=text_to_label,
        split_name="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False
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

    best_val_acc = 0.0
    best_path = "checkpoints_text_cls/best_model_split.pth"

    os.makedirs("checkpoints_text_cls", exist_ok=True)
    
    log_dir = "runs/m2voice_text_cls_split"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")

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

            total_loss += loss.item()

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

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        print("\n========== Epoch Summary ==========")
        print(f"Epoch      : {epoch + 1}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc * 100:.2f}%")

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            label_texts,
            device,
            name="val"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "label_texts": label_texts,
                    "best_val_acc": best_val_acc,
                    "config": {
                        "hidden_size": config.hidden_size,
                        "mlp_dim": config.mlp_dim,
                        "num_heads": config.num_heads,
                        "num_layers": config.num_layers,
                        "dropout": config.dropout,
                        "target_seq_len": target_seq_len,
                    }
                },
                best_path
            )

            print(f"Saved best model: {best_path}")
            print(f"Best Val Acc = {best_val_acc * 100:.2f}%")

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1)

    print("\n========== Load Best Model and Test ==========")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluate(
        model,
        test_loader,
        label_texts,
        device,
        name="test"
    )

    print(f"\nBest Val Acc: {best_val_acc * 100:.2f}%")
    writer.close()


if __name__ == "__main__":
    train()
    
    