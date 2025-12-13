from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from data import TokenNPYDataset
from model import GPT, GPTConfig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon support (optional)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/tokenized/manifest.parquet")
    ap.add_argument("--out_dir", type=str, default="runs/exp1")
    ap.add_argument("--seq_len", type=int, default=1024)  # input length (labels is seq_len-1)
    ap.add_argument("--vocab_size", type=int, default=1412)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    train_ds = TokenNPYDataset(args.manifest, split="train", seq_len=args.seq_len, random_crop=True, seed=args.seed)
    val_ds = TokenNPYDataset(args.manifest, split="val", seq_len=args.seq_len, random_crop=False, seed=args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=max(0, args.num_workers // 2), pin_memory=(device.type == "cuda")
    )

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len - 1,  # because dataset returns input_ids of length seq_len-1
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    step = 0
    t0 = time()
    model.train()

    while step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            step += 1

            if step % 50 == 0:
                dt = time() - t0
                print(f"step {step}/{args.max_steps} | loss {loss.item():.4f} | {dt:.1f}s")
                t0 = time()

            if step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"[eval] step {step} | val_loss {val_loss:.4f}")

            if step % args.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                }
                torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")
                torch.save(ckpt, out_dir / "ckpt_latest.pt")
                print(f"[save] {out_dir/'ckpt_latest.pt'}")

            if step >= args.max_steps:
                break

    print("Done.")


if __name__ == "__main__":
    main()
