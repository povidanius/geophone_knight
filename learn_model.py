#!/usr/bin/env python3
"""Train the compact still/walk network from collected .npz sessions."""

import argparse
from pathlib import Path
import random

import numpy as np

from motion_model import (CLASSES, MODEL_SAMPLES, WINDOW_SECONDS,
                          build_network, prepare_window)


def load_sessions(data_dir):
    sessions = []
    for path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(path) as data:
            label = str(data["label"].item())
            if label not in CLASSES:
                continue
            sessions.append((path, np.asarray(data["samples"], np.float32),
                             float(data["fs"]), CLASSES.index(label)))
    return sessions


def windows(session, window_s, stride_s=0.10):
    _, samples, fs, label = session
    width = max(8, int(round(fs * window_s)))
    stride = max(1, int(round(fs * stride_s)))
    return [(prepare_window(samples[:end], fs, window_s=window_s), label)
            for end in range(width, len(samples) + 1, stride)]


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).with_name("training_data")))
    ap.add_argument("--output",
                    default=str(Path(__file__).with_name("geophone_model.pt")))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                    help="Signal history used by both training and inference.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be greater than zero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sessions = load_sessions(args.data_dir)
    counts = {name: sum(s[3] == i for s in sessions)
              for i, name in enumerate(CLASSES)}
    if any(v < 1 for v in counts.values()):
        raise SystemExit("Need at least 1 recording per class; got %r" % counts)

    # Split whole recording sessions, never overlapping windows, to avoid
    # optimistic validation leakage.
    train_sessions, val_sessions = [], []
    temporal_fallback = False
    for label in range(len(CLASSES)):
        group = [s for s in sessions if s[3] == label]
        random.shuffle(group)
        if len(group) == 1:
            temporal_fallback = True
            path, samples, fs, class_id = group[0]
            width = max(8, int(round(fs * args.window_seconds)))
            # With only one session, split it in time and leave a complete
            # window-sized gap on both sides. A middle validation block lets
            # training see normal signal drift near both ends of the session.
            val_start = int(len(samples) * 0.40)
            val_end = int(len(samples) * 0.60)
            left = samples[:max(0, val_start - width)]
            validation = samples[val_start:val_end]
            right = samples[min(len(samples), val_end + width):]
            if (len(left) < width or len(right) < width
                    or len(validation) < width):
                raise SystemExit(
                    "%s recording is too short for a safe train/validation "
                    "split; collect at least %.1f seconds."
                    % (CLASSES[label], 6.0 * args.window_seconds))
            train_sessions.append((path, left, fs, class_id))
            train_sessions.append((path, right, fs, class_id))
            val_sessions.append((path, validation, fs, class_id))
            print("Only one %s session: using separated time regions for "
                  "validation." % CLASSES[label])
        else:
            nval = max(1, int(round(len(group) * 0.2)))
            val_sessions.extend(group[:nval])
            train_sessions.extend(group[nval:])
    train = [item for s in train_sessions
             for item in windows(s, args.window_seconds)]
    val = [item for s in val_sessions
           for item in windows(s, args.window_seconds)]
    random.shuffle(train)

    def dataset(items):
        x = torch.from_numpy(np.stack([v[0] for v in items]))
        y = torch.tensor([v[1] for v in items], dtype=torch.long)
        return TensorDataset(x, y)

    train_data = dataset(train)
    train_labels = train_data.tensors[1]
    class_counts = torch.bincount(train_labels, minlength=len(CLASSES)).float()
    sample_weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=sampler)
    val_x, val_y = dataset(val).tensors
    model = build_network()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state, best_epoch = -1.0, None, 1
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            # Mild sensor-noise augmentation improves robustness without
            # erasing useful absolute-amplitude information.
            x = x + torch.randn_like(x) * 0.015
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred = model(val_x).argmax(1)
            acc = float((pred == val_y).float().mean())
            recalls = []
            for class_id in range(len(CLASSES)):
                mask = val_y == class_id
                recalls.append(float((pred[mask] == class_id).float().mean()))
            balanced_acc = float(np.mean(recalls))
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        recall_text = "  ".join(
            "%s %d%%" % (name, round(100 * value))
            for name, value in zip(CLASSES, recalls))
        print("epoch %3d  val %.1f%%  balanced %.1f%%  recall %s" %
              (epoch, 100 * acc, 100 * balanced_acc, recall_text))

    # Validation selects the useful epoch count. When temporal fallback was
    # necessary, refit a fresh model on every recorded window so the deployed
    # classifier does not throw away the held-out middle of the only session.
    if temporal_fallback:
        all_items = [item for s in sessions
                     for item in windows(s, args.window_seconds)]
        all_data = dataset(all_items)
        all_labels = all_data.tensors[1]
        all_counts = torch.bincount(
            all_labels, minlength=len(CLASSES)).float()
        all_weights = (1.0 / all_counts)[all_labels]
        all_sampler = WeightedRandomSampler(
            all_weights, len(all_weights), replacement=True)
        all_loader = DataLoader(all_data, batch_size=args.batch_size,
                                sampler=all_sampler)
        model = build_network()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                weight_decay=1e-3)
        for _ in range(best_epoch):
            model.train()
            for x, y in all_loader:
                x = x + torch.randn_like(x) * 0.015
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        best_state = model.state_dict()
        print("Refitted deployment model on all %d windows for %d epochs." %
              (len(all_items), best_epoch))

    torch.save({"state_dict": best_state, "classes": CLASSES,
                "model_samples": MODEL_SAMPLES,
                "window_seconds": args.window_seconds,
                "validation_accuracy": best_acc}, args.output)
    print("Saved %s (best validation accuracy %.1f%%)." %
          (args.output, 100 * best_acc))


if __name__ == "__main__":
    main()
