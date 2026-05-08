# Generic training loop with mixed precision support
# Used across all models — scratch CNN, EfficientNet, ResNet, AST

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

import wandb


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scaler=None, clip_grad=None, accum_steps=1):
    # Train for one epoch with optional mixed precision and gradient accumulation
    model.train()
    total_loss, n = 0, 0

    optimizer.zero_grad()
    for step, (mel, labels) in enumerate(tqdm(train_loader, leave=False)):
        mel = mel.to(device)
        labels = labels.to(device)

        if scaler:
            with autocast():
                logits = model(mel)
                loss = criterion(logits, labels) / accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * len(labels) * accum_steps
        n += len(labels)

    return total_loss / n


@torch.no_grad()
def evaluate(model, val_loader, device):
    # Evaluate model and return macro F1 and accuracy
    model.eval()
    preds, labs = [], []
    for mel, lb in val_loader:
        mel = mel.to(device)
        preds.extend(model(mel).argmax(1).cpu().numpy())
        labs.extend(lb.numpy() if isinstance(lb, torch.Tensor) else lb)
    f1 = f1_score(labs, preds, average='macro')
    acc = np.mean(np.array(preds) == np.array(labs))
    return f1, acc


def train_model(model, train_dataset_fn, val_loader, criterion, optimizer,
                scheduler, device, epochs, output_dir, model_name,
                use_amp=False, clip_grad=None, accum_steps=1,
                batch_size=32, num_workers=2):
    # Full training loop
    # train_dataset_fn: callable returning fresh dataset each epoch (on-the-fly augmentation)
    # val_loader: DataLoader for validation
    # model_name: name for saving weights and wandb
    scaler = GradScaler() if use_amp else None
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # fresh dataset each epoch for on-the-fly augmentation
        train_ds = train_dataset_fn()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )

        # train
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, clip_grad=clip_grad, accum_steps=accum_steps
        )
        scheduler.step()

        # evaluate
        f1, acc = evaluate(model, val_loader, device)

        # save best
        tag = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_{model_name}.pth'))
            tag = ""

        # log
        wandb.log({
            "epoch": epoch, "loss": avg_loss,
            "val_f1": f1, "val_acc": acc,
            "lr": optimizer.param_groups[0]['lr']
        })
        elapsed = time.time() - t0
        print(f"E{epoch:02d}/{epochs} | loss={avg_loss:.4f} | f1={f1:.4f} | acc={acc:.4f} | {elapsed:.0f}s{tag}")

    print(f"\nBest val F1: {best_f1:.4f}")
    return best_f1