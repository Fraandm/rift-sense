"""Entrena DraftNet (valoracion de draft) y guarda metricas + checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..utils import read_json, write_json
from .draft_net import DraftNet

PROCESSED = config.PROCESSED_DIR


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n0 == 0 or n1 == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(np.float64)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    # Calibracion en 10 bins
    bins = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum() >= 20:
            bins.append(
                {
                    "lo": round(float(lo), 2),
                    "hi": round(float(hi), 2),
                    "n": int(sel.sum()),
                    "pred": round(float(p[sel].mean()), 4),
                    "actual": round(float(y[sel].mean()), 4),
                }
            )
    return {
        "logloss": round(logloss, 5),
        "auc": round(auc_score(y, p), 5),
        "accuracy": round(float((pred == y).mean()), 5),
        "brier": round(brier, 5),
        "n": int(len(y)),
        "calibration": bins,
    }


def build_tokens(data: dict, unk: int) -> tuple[torch.Tensor, ...]:
    picks = torch.from_numpy(data["picks"].astype(np.int64))
    bans = torch.from_numpy(data["bans"].astype(np.int64))
    pick_roles = torch.from_numpy(data["pick_roles"].astype(np.int64))
    pick_side = torch.from_numpy(data["pick_side"].astype(np.int64))
    ban_sides = torch.from_numpy(data["ban_sides"].astype(np.int64))

    n = picks.shape[0]
    champ = torch.cat([picks, bans], dim=1)
    role = torch.cat([pick_roles, torch.zeros(n, 10, dtype=torch.long)], dim=1)
    side = torch.cat([pick_side, ban_sides], dim=1)
    kind = torch.cat([torch.ones(n, 10, dtype=torch.long), torch.full((n, 10), 2, dtype=torch.long)], dim=1)
    mask = champ > 0
    mask[:, :10] = True
    champ = champ.clone()
    champ[~mask] = 0
    return champ, role, side, kind, mask


def augment(
    champ: torch.Tensor,
    role: torch.Tensor,
    side: torch.Tensor,
    kind: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    unk: int,
    *,
    p_mask: float = 0.15,
    p_swap: float = 0.5,
) -> None:
    """Enmascara picks (draft parcial) e intercambia equipos in-place."""
    b = champ.shape[0]
    drop = torch.rand(b, 10, device=champ.device) < p_mask
    champion_part = champ[:, :10]
    champion_part[drop] = unk

    swap = torch.rand(b, device=champ.device) < p_swap
    if swap.any():
        idx = torch.cat(
            [
                torch.arange(5, 10),
                torch.arange(0, 5),
                torch.arange(15, 20),
                torch.arange(10, 15),
            ]
        )
        champ[swap] = champ[swap][:, idx]
        role[swap] = role[swap][:, idx]
        side[swap] = 1 - side[swap]
        kind[swap] = kind[swap][:, idx]
        mask[swap] = mask[swap][:, idx]
        y[swap] = 1 - y[swap]


@torch.no_grad()
def predict(model: nn.Module, tensors: tuple, idx: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    champ, role, side, kind, mask = tensors
    out = []
    for start in range(0, len(idx), batch_size):
        sel = torch.from_numpy(idx[start : start + batch_size].astype(np.int64))
        logits = model(champ[sel], role[sel], side[sel], kind[sel], mask[sel])
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la red de draft")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mask-prob", type=float, default=0.08, help="Prob. de ocultar un pick (draft parcial)")
    parser.add_argument("--swap-prob", type=float, default=0.0, help="Prob. de intercambiar equipos")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", default=str(config.DRAFT_MODEL_PATH))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, int((torch.get_num_threads() or 4))))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = read_json(PROCESSED / "vocab.json")
    data = np.load(PROCESSED / "dataset.npz")
    n_champions = len(vocab["championList"])
    unk = n_champions + 1

    champ, role, side, kind, mask = build_tokens(data, unk)
    y = torch.from_numpy(data["y"].astype(np.float32))
    split = data["split"]
    dates = data["date"].astype(np.float64)

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]

    # Peso por recencia: el meta actual importa mas
    max_date = dates.max()
    days_ago = max_date - dates
    weights = 0.2 + 0.8 * np.exp(-days_ago / 365.0)
    w_train = weights[train_idx]
    w_train = w_train / w_train.mean()
    w_tensor = torch.from_numpy(w_train.astype(np.float32))

    champ = champ.to(device)
    role = role.to(device)
    side = side.to(device)
    kind = kind.to(device)
    mask = mask.to(device)
    y = y.to(device)
    w_tensor = w_tensor.to(device)
    tensors = (champ, role, side, kind, mask)

    model = DraftNet(
        n_champions=n_champions,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_idx_t = torch.from_numpy(train_idx.astype(np.int64))
    best = {"val_auc": -1.0, "val_logloss": math.inf, "epoch": -1, "state": None, "val_metrics": None}
    history = []
    print(
        f"Entrenando DraftNet en {device}: {n_params:,} parametros | "
        f"{len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test"
    )
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx_t))
        total_loss = 0.0
        seen = 0
        t0 = time.time()
        for start in range(0, len(perm), args.batch_size):
            batch_idx = train_idx_t[perm[start : start + args.batch_size]]
            c = champ[batch_idx].clone()
            r = role[batch_idx].clone()
            s = side[batch_idx].clone()
            k = kind[batch_idx].clone()
            m = mask[batch_idx].clone()
            target = y[batch_idx].clone()
            w = w_tensor[perm[start : start + args.batch_size]]

            augment(
                c,
                r,
                s,
                k,
                m,
                target,
                unk,
                p_mask=args.mask_prob,
                p_swap=args.swap_prob,
            )
            logits = model(c, r, s, k, m)
            loss = (loss_fn(logits, target) * w).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()) * len(batch_idx)
            seen += len(batch_idx)
            if (start // args.batch_size) % 40 == 0:
                print(
                    f"  epoch {epoch} | {seen}/{len(perm)} | loss {total_loss / max(seen,1):.4f}",
                    flush=True,
                )
        scheduler.step()
        val_probs = predict(model, tensors, val_idx)
        val_m = metrics(y.cpu().numpy()[val_idx], val_probs)
        train_loss = total_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_logloss": val_m["logloss"],
                "val_auc": val_m["auc"],
                "val_accuracy": val_m["accuracy"],
                "lr": round(opt.param_groups[0]["lr"], 8),
                "seconds": round(time.time() - t0, 1),
            }
        )
        improved = val_m["auc"] > best["val_auc"] + 1e-5
        marker = "*" if improved else ""
        print(
            f"epoch {epoch:02d}{marker} train {train_loss:.4f} | "
            f"val logloss {val_m['logloss']:.4f} auc {val_m['auc']:.4f} acc {val_m['accuracy']:.4f} "
            f"({time.time()-t0:.0f}s)"
        )
        if improved:
            best = {
                "val_auc": val_m["auc"],
                "val_logloss": val_m["logloss"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "val_metrics": val_m,
            }
        elif epoch - best["epoch"] >= args.patience:
            print(f"Early stopping en epoch {epoch}")
            break

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best["epoch"] = int(history[-1]["epoch"]) if history else 0

    model.load_state_dict(best["state"])
    model.eval()

    val_probs = predict(model, tensors, val_idx)
    test_probs = predict(model, tensors, test_idx)
    val_m = metrics(y.cpu().numpy()[val_idx], val_probs)
    test_m = metrics(y.cpu().numpy()[test_idx], test_probs)

    # Rendimiento con draft parcial (2 picks desconocidos por equipo)
    partial = {}
    rng = np.random.default_rng(args.seed)
    for n_unknown in (2, 4):
        probs_all = []
        for _ in range(3):
            c = champ[torch.from_numpy(test_idx.astype(np.int64))].clone()
            drop = torch.from_numpy(rng.random((len(test_idx), 10)) < (n_unknown / 10)).to(device)
            c[:, :10][drop] = unk
            with torch.no_grad():
                logits = model(c, role[torch.from_numpy(test_idx.astype(np.int64))], side[torch.from_numpy(test_idx.astype(np.int64))],
                               kind[torch.from_numpy(test_idx.astype(np.int64))], mask[torch.from_numpy(test_idx.astype(np.int64))])
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
        partial[f"{n_unknown}_unknown"] = metrics(y.cpu().numpy()[test_idx], np.mean(probs_all, axis=0))

    ckpt = {
        "state_dict": best["state"],
        "n_champions": n_champions,
        "d_model": args.d_model,
        "n_heads": args.heads,
        "n_layers": args.layers,
        "dropout": args.dropout,
        "unk_index": unk,
        "ddragonVersion": vocab.get("ddragonVersion"),
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": best["epoch"],
        "val": val_m,
        "test": test_m,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    report = {
        "model": "DraftNet",
        "task": "P(gana el equipo del lado azul) a partir de picks y bans",
        "trainedAt": ckpt["trainedAt"],
        "device": device,
        "parameters": n_params,
        "epochsRun": len(history),
        "bestEpoch": best["epoch"],
        "trainingMinutes": round((time.time() - started) / 60, 2),
        "samples": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "dateRange": [str(data["date"].min()), str(data["date"].max())],
        "val": val_m,
        "test": test_m,
        "partialDraft": partial,
        "history": history,
        "config": {
            "d_model": args.d_model,
            "n_heads": args.heads,
            "n_layers": args.layers,
            "dropout": args.dropout,
            "batch": args.batch_size,
            "lr": args.lr,
            "maskProb": args.mask_prob,
            "swapProb": args.swap_prob,
        },
    }
    write_json(config.DRAFT_REPORT_PATH, report)
    print(
        f"Guardado {out_path}\n"
        f"TEST: logloss {test_m['logloss']:.4f} | auc {test_m['auc']:.4f} | acc {test_m['accuracy']:.4f} | "
        f"n {test_m['n']:,}"
    )
    print(f"Reporte -> {config.DRAFT_REPORT_PATH}")


if __name__ == "__main__":
    main()
