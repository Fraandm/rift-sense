"""Entrena BuildNet (siguiente objeto) y guarda metricas + checkpoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..utils import parse_patch, read_json, write_json
from .build_net import BuildNet

PROCESSED = config.PROCESSED_DIR


def patch_code(value: str) -> int:
    major, minor = parse_patch(value)
    return major * 100 + minor


def filter_rows(data: dict, min_patch: str) -> dict[str, np.ndarray]:
    patch_by_game = data["patch"]
    row_patch = patch_by_game[data["b_game"]]
    keep = row_patch >= patch_code(min_patch)
    split = data["split"][data["b_game"]]
    return {
        "champ": data["b_champ"][keep],
        "role": data["b_role"][keep],
        "side": data["b_side"][keep],
        "ally": data["b_ally"][keep],
        "enemy": data["b_enemy"][keep],
        "items": data["b_items"][keep],
        "length": data["b_len"][keep],
        "split": split[keep],
        "patch": row_patch[keep],
    }


def make_tensors(rows: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    def t(key: str, dtype=torch.long):
        return torch.from_numpy(rows[key].astype(np.int64)).to(device)

    return {
        "champ": t("champ"),
        "role": t("role"),
        "side": t("side"),
        "ally": t("ally"),
        "enemy": t("enemy"),
        "items": t("items"),
        "length": t("length"),
        "split": rows["split"],
    }


def evaluate(model: BuildNet, tensors: dict, indices: np.ndarray, n_items: int) -> dict:
    model.eval()
    device = tensors["champ"].device
    top1 = top3 = top5 = 0
    mrr = 0.0
    total = 0
    per_role = {i: [0, 0] for i in range(1, 6)}
    lengths_np = tensors["length"].cpu().numpy()
    with torch.no_grad():
        for k in range(7):
            sel_np = indices[lengths_np[indices] > k]
            if len(sel_np) == 0:
                continue
            sel = torch.from_numpy(sel_np.astype(np.int64)).to(device)
            logits = model(
                tensors["champ"][sel],
                tensors["role"][sel],
                tensors["side"][sel],
                tensors["ally"][sel],
                tensors["enemy"][sel],
                tensors["items"][sel],
                torch.full_like(sel, k),
            )
            target = tensors["items"][sel, k]
            probs = torch.softmax(logits, dim=-1)
            ranks = torch.argsort(probs, dim=-1, descending=True)
            pos = (ranks == target.unsqueeze(-1)).float().argmax(dim=-1)
            top1 += int((pos == 0).sum())
            top3 += int((pos < 3).sum())
            top5 += int((pos < 5).sum())
            mrr += float((1.0 / (pos.float() + 1.0)).sum())
            roles = tensors["role"][sel]
            for role_id in per_role:
                mask = roles == role_id
                if mask.any():
                    per_role[role_id][0] += int((pos[mask] == 0).sum())
                    per_role[role_id][1] += int(mask.sum())
            total += len(sel_np)

    return {
        "n": total,
        "top1": round(top1 / max(total, 1), 5),
        "top3": round(top3 / max(total, 1), 5),
        "top5": round(top5 / max(total, 1), 5),
        "mrr": round(mrr / max(total, 1), 5),
        "perRoleTop1": {
            config.ROLES[r - 1]: round(v[0] / max(v[1], 1), 5) for r, v in per_role.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la red de builds")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-patch", default=config.BUILD_MIN_PATCH)
    parser.add_argument("--max-train", type=int, default=0, help="Limita filas de train (0 = todas)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", default=str(config.BUILD_MODEL_PATH))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = read_json(PROCESSED / "vocab.json")
    n_champions = len(vocab["championList"])
    n_items = len(vocab["itemList"])
    data = np.load(PROCESSED / "dataset.npz")
    rows = filter_rows(data, args.min_patch)
    tensors = make_tensors(rows, device)

    train_idx = np.where(rows["split"] == 0)[0]
    val_idx = np.where(rows["split"] == 1)[0]
    test_idx = np.where(rows["split"] == 2)[0]
    if args.max_train and len(train_idx) > args.max_train:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)

    # Baseline de popularidad (objeto mas comprado)
    flat = rows["items"][train_idx][rows["items"][train_idx] > 0]
    counts = np.bincount(flat, minlength=n_items + 1)
    top_items = np.argsort(counts)[::-1][:5]
    baseline_top1 = counts[top_items[0]] / max(counts[1:].sum(), 1)

    model = BuildNet(n_champions=n_champions, n_items=n_items).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)

    print(
        f"Entrenando BuildNet en {device}: {n_params:,} parametros | "
        f"{len(train_idx):,} train / {len(val_idx):,} val / {len(test_idx):,} test "
        f"| min patch {args.min_patch}"
    )

    all_items = tensors["items"]
    lengths = tensors["length"].cpu().numpy()
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(train_idx))
        total_loss = 0.0
        seen = 0
        t0 = time.time()
        for start in range(0, len(perm), args.batch_size):
            batch_rows = train_idx[perm[start : start + args.batch_size]]
            sel = torch.from_numpy(batch_rows.astype(np.int64)).to(device)
            lens = torch.from_numpy(lengths[batch_rows]).to(device)
            k = (torch.rand(len(batch_rows), device=device) * lens).long()
            logits = model(
                tensors["champ"][sel],
                tensors["role"][sel],
                tensors["side"][sel],
                tensors["ally"][sel],
                tensors["enemy"][sel],
                all_items[sel],
                k,
            )
            target = all_items[sel, k]
            loss = loss_fn(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()) * len(batch_rows)
            seen += len(batch_rows)
            if (start // args.batch_size) % 60 == 0:
                print(f"  epoch {epoch} | {seen}/{len(perm)} | loss {total_loss/max(seen,1):.4f}", flush=True)

        scheduler.step()
        val_m = evaluate(model, tensors, val_idx, n_items)
        train_loss = total_loss / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_top1": val_m["top1"],
                "val_top3": val_m["top3"],
                "val_mrr": val_m["mrr"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        improved = train_loss < best["val_loss"] or best["state"] is None
        # Seleccion por top3, mas estable que la loss para ranking
        if best["state"] is not None:
            improved = val_m["top3"] > best.get("val_top3", 0) + 1e-6
        marker = "*" if improved else ""
        print(
            f"epoch {epoch:02d}{marker} train {train_loss:.4f} | val top1 {val_m['top1']:.3f} "
            f"top3 {val_m['top3']:.3f} mrr {val_m['mrr']:.3f} ({time.time()-t0:.0f}s)"
        )
        if improved:
            best = {
                "val_loss": train_loss,
                "val_top3": val_m["top3"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= args.patience:
            print(f"Early stopping en epoch {epoch}")
            break

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best["epoch"] = int(history[-1]["epoch"]) if history else 0

    model.load_state_dict(best["state"])
    model.eval()
    val_m = evaluate(model, tensors, val_idx, n_items)
    test_m = evaluate(model, tensors, test_idx, n_items)
    test_m["popularityBaselineTop1"] = round(float(baseline_top1), 5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state"],
            "n_champions": n_champions,
            "n_items": n_items,
            "d_model": 96,
            "hidden": 192,
            "dropout": 0.15,
            "minPatch": args.min_patch,
            "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": best["epoch"],
            "val": val_m,
            "test": test_m,
            "ddragonVersion": vocab.get("ddragonVersion"),
        },
        out_path,
    )

    report = {
        "model": "BuildNet",
        "task": "Siguiente objeto dado el contexto y el historial de compras",
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "parameters": n_params,
        "minPatch": args.min_patch,
        "epochsRun": len(history),
        "bestEpoch": best["epoch"],
        "trainingMinutes": round((time.time() - started) / 60, 2),
        "samples": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "itemVocabulary": n_items,
        "val": val_m,
        "test": test_m,
        "popularityBaselineTop1": round(float(baseline_top1), 5),
        "topItems": [
            {"idx": int(i), "id": vocab["itemList"][int(i) - 1]["id"], "name": vocab["itemList"][int(i) - 1]["name"]}
            for i in top_items
            if i > 0
        ],
        "history": history,
        "config": {"d_model": 96, "hidden": 192, "batch": args.batch_size, "lr": args.lr},
    }
    write_json(config.BUILD_REPORT_PATH, report)
    print(
        f"Guardado {out_path}\n"
        f"TEST: top1 {test_m['top1']:.3f} | top3 {test_m['top3']:.3f} | top5 {test_m['top5']:.3f} | "
        f"baseline pop. {baseline_top1:.3f}"
    )
    print(f"Reporte -> {config.BUILD_REPORT_PATH}")


if __name__ == "__main__":
    main()
