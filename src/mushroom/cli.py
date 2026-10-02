"""Command-line entry point: ``uv run mushroom {data,run,figures,report,all}``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mushroom", description=__doc__)
    parser.add_argument("step", choices=["data", "run", "figures", "report", "all"])
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    # Imports are deferred so `mushroom data` does not pay for scikit-learn/plotly.
    if args.step in {"data", "all"}:
        from mushroom.data import build_processed

        df = build_processed(force_download=args.force_download)
        print(f"data: {len(df):,} specimens x {df.shape[1] - 1} variables")
    if args.step in {"run", "all"}:
        from mushroom.pipeline import run

        run()
    if args.step in {"figures", "run", "all"}:
        from mushroom.figures import build as build_figures

        build_figures()
        print("figures: docs/figures/")
    if args.step in {"report", "all"}:
        from mushroom.report import build as build_report

        build_report()
        print("report: site/index.html")


if __name__ == "__main__":
    main()
