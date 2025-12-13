#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from miditok import REMI, TokenizerConfig

from model import GPT, GPTConfig


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer_from_tokenized_dir(tokenized_dir: Path) -> tuple[REMI, List[str], dict, int, dict]:
    """
    Returns:
      tokenizer (REMI)
      special_tokens (list[str])
      special_tok2id (dict[str,int])
      offset (int)
      genres_map (dict[str, int])  # genre string -> special token id (pre-dataset-shift)
    """
    cfgj = json.loads((tokenized_dir / "tokenizer_config.json").read_text())
    special_tokens = json.loads((tokenized_dir / "special_tokens.json").read_text())
    special_tok2id = json.loads((tokenized_dir / "special_token_to_id.json").read_text())
    offset = json.loads((tokenized_dir / "offset.json").read_text())["offset"]

    # rebuild TokenizerConfig
    cfg = TokenizerConfig(
        beat_res={(0, 4): int(cfgj["time_res"])},
        use_velocities=True,
        num_velocities=int(cfgj["velocity_bins"]),
        use_note_duration=True,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        max_bar_embedding=int(cfgj["max_bar_embedding"]),
    )
    tokenizer = REMI(cfg)

    # map genre -> id (pre-dataset-shift)
    genres_map = {}
    for tok, tid in special_tok2id.items():
        if tok.startswith("<GENRE=") and tok.endswith(">"):
            g = tok[len("<GENRE=") : -1]
            genres_map[g] = int(tid)

    return tokenizer, special_tokens, special_tok2id, int(offset), genres_map


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    """
    logits: (vocab,)
    """
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # sort
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=0)

    # keep smallest set with cum <= top_p
    mask = cum <= top_p
    # ensure at least one token
    if not torch.any(mask):
        mask[0] = True

    kept_probs = sorted_probs[mask]
    kept_idx = sorted_idx[mask]
    kept_probs = kept_probs / kept_probs.sum()

    choice = torch.multinomial(kept_probs, num_samples=1).item()
    return int(kept_idx[choice].item())


@torch.no_grad()
def generate_tokens(
    model: GPT,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[int]:
    """
    prompt_ids: model-token-space ids (where PAD=0, and everything else already includes +1 dataset shift)
    returns: full sequence in model-token-space ids
    """
    device = next(model.parameters()).device
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]  # (1,T)

    for _ in range(max_new_tokens):
        # crop to block size
        if x.size(1) > model.cfg.block_size:
            x = x[:, -model.cfg.block_size :]

        logits, _ = model(x, labels=None)          # (1,T,V)
        next_logits = logits[0, -1, :]             # (V,)

        next_id = top_p_sample(next_logits, top_p=top_p, temperature=temperature)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    return x[0].tolist()


def decode_to_midi(
    tokenizer: REMI,
    ids_model_space: List[int],
    offset: int,
    out_mid_path: Path,
) -> None:
    """
    Convert model-space ids back to MidiTok base ids and decode to MIDI.
    """
    # Remove PAD
    ids_model_space = [i for i in ids_model_space if i != 0]

    # Undo dataset shift (+1 during training dataset)
    ids_tok_space = [i - 1 for i in ids_model_space if i - 1 >= 0]

    if len(ids_tok_space) < 3:
        raise ValueError("Generated sequence too short to decode.")

    # Strip leading special tokens (<GENRE=...>, <BOS>, etc.)
    i = 0
    while i < len(ids_tok_space) and ids_tok_space[i] < offset:
        i += 1
    content = ids_tok_space[i:]

    # Undo +offset (keep only valid)
    base_ids = [j - offset for j in content if (j - offset) >= 0]

    if len(base_ids) == 0:
        raise ValueError("No base ids left after stripping special tokens.")

    # MidiTok v3: decode from ids directly
    # Depending on exact build, tokenizer.decode(ids) returns a symusic.Score
    if hasattr(tokenizer, "decode"):
        score = tokenizer.decode(base_ids)
    else:
        # fallback (rare): tokenizer(base_ids) might decode
        score = tokenizer(base_ids)

    score.dump_midi(str(out_mid_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenized_dir", type=str, default="data/tokenized")
    ap.add_argument("--out", type=str, default="out/generated.mid")

    ap.add_argument("--genre", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenized_dir = Path(args.tokenized_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, special_tokens, special_tok2id, offset, genres_map = load_tokenizer_from_tokenized_dir(tokenized_dir)

    if args.genre not in genres_map:
        available = ", ".join(sorted(genres_map.keys()))
        raise ValueError(f"Unknown genre '{args.genre}'. Available: {available}")

    device = get_device()
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Build prompt: [<GENRE=...>, <BOS>] in token-space, then +1 shift to model-space
    genre_tok = f"<GENRE={args.genre}>"
    genre_id = int(special_tok2id[genre_tok])
    bos_id = int(special_tok2id[BOS_TOKEN])

    prompt_tok_space = [genre_id, bos_id]
    prompt_model_space = [x + 1 for x in prompt_tok_space]  # dataset shift (+1)

    ids_model = generate_tokens(
        model=model,
        prompt_ids=prompt_model_space,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    decode_to_midi(
        tokenizer=tokenizer,
        ids_model_space=ids_model,
        offset=offset,
        out_mid_path=out_path,
    )

    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()
