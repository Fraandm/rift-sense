"""Ejecuta el pipeline completo: estáticos -> scraping -> dataset -> modelos."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, args: list[str], *, optional: bool = False) -> bool:
    print(f"\n=== {name} ===")
    started = time.time()
    result = subprocess.run([sys.executable, "-m", *args], cwd=ROOT)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"[{name}] fallo con codigo {result.returncode} ({elapsed:.0f}s)")
        if optional:
            return False
        raise SystemExit(result.returncode)
    print(f"[{name}] OK en {elapsed/60:.1f} min")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo de RiftSense")
    parser.add_argument("--since", default="2024-01-01")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-static", action="store_true")
    args = parser.parse_args()

    static_args = ["lolanalyzer.scraper.ddragon"]
    if args.force_static:
        static_args.append("--force")
    run("Datos estaticos", static_args)

    if not args.skip_scrape:
        run("Partidas profesionales", ["lolanalyzer.scraper.leaguepedia", "--since", args.since])

    run("Dataset y priors", ["lolanalyzer.data.build_dataset"])

    if not args.skip_train:
        run("Entrenar DraftNet", ["lolanalyzer.models.train_draft", "--epochs", "40"])
        run("Entrenar BuildNet", ["lolanalyzer.models.train_build", "--epochs", "14"])

    print("\nPipeline completado. Arranca la app con:")
    print("  python -m lolanalyzer.api.server")


if __name__ == "__main__":
    main()
