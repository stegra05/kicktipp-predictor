import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

import numpy as np
import pandas as pd
import sys

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# Ensure src/ is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_seasons_arg(seasons_arg: str | None, cache_dir: Path, current_season: int) -> Tuple[int, int]:
    """Parse seasons range from CLI or infer from cache.

    Accepts formats like:
      - "2020-2024"
      - "2022" (single season)
    If None, tries to infer from cache files; otherwise defaults to (current_season-5 .. current_season).
    """
    if seasons_arg:
        s = seasons_arg.strip()
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                start, end = int(a), int(b)
            except Exception:
                raise ValueError(f"Invalid --seasons value: {seasons_arg}")
            if start > end:
                start, end = end, start
            return start, end
        try:
            single = int(s)
            return single, single
        except Exception:
            raise ValueError(f"Invalid --seasons value: {seasons_arg}")

    # Infer from cache files like matches_2024.pkl
    seasons: List[int] = []
    if cache_dir.exists():
        for p in cache_dir.glob("matches_*.pkl"):
            try:
                seasons.append(int(p.stem.split("_")[-1]))
            except Exception:
                continue
    if seasons:
        return min(seasons), max(seasons)

    # Fallback: last 5 seasons ending at current
    return max(2000, current_season - 5), current_season


def _detect_feature_columns(df: pd.DataFrame, extra_exclude: Iterable[str] | None = None) -> List[str]:
    """Return numeric feature columns excluding meta/target columns."""
    default_exclude = {
        "match_id",
        "matchday",
        "date",
        "home_team",
        "away_team",
        "is_finished",
        "home_score",
        "away_score",
        "goal_difference",
        "result",
    }
    if extra_exclude:
        default_exclude.update(extra_exclude)

    num_df = df.select_dtypes(include=[np.number, bool])
    feature_cols = [c for c in num_df.columns if c not in default_exclude]
    return feature_cols


def _build_correlation_groups(df: pd.DataFrame, feature_cols: List[str], threshold: float) -> Tuple[Dict[str, str], List[Tuple[str, str, float]]]:
    """Return mapping col->group_rep and list of correlated pairs above threshold.

    Uses a union-find to group columns connected by corr > threshold.
    """
    if not feature_cols:
        return {}, []

    # Fill NaNs with column means to avoid artificial correlation by zeros
    work = df[feature_cols].copy()
    for c in work.columns:
        if work[c].isna().any():
            work[c] = work[c].astype(float).fillna(work[c].mean())

    corr = work.corr().abs()

    # Gather upper-triangular pairs above threshold
    pairs: List[Tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iloc[i, j])
            if v > threshold:
                pairs.append((cols[i], cols[j], v))

    if not pairs:
        return {c: c for c in feature_cols}, []

    # Union-Find
    parent: Dict[str, str] = {c: c for c in feature_cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _ in pairs:
        union(a, b)

    # Build components
    components: Dict[str, Set[str]] = {}
    for c in feature_cols:
        r = find(c)
        components.setdefault(r, set()).add(c)

    # Choose representative per group using missingness, variance, lexicographic
    representatives: Dict[str, str] = {}
    missing_counts = {c: int(df[c].isna().sum()) for c in feature_cols}
    variances = {c: float(pd.Series(df[c], copy=False).astype(float).var(ddof=1)) for c in feature_cols}

    for root, members in components.items():
        if len(members) == 1:
            col = next(iter(members))
            representatives[col] = col
            continue
        candidates = sorted(
            list(members),
            key=lambda c: (
                missing_counts.get(c, 0),
                -variances.get(c, 0.0),
                c,
            ),
        )
        keep = candidates[0]
        for m in members:
            representatives[m] = keep

    return representatives, pairs


def _select_columns_by_groups(representatives: Dict[str, str]) -> Tuple[List[str], List[str]]:
    keep: Set[str] = set()
    drop: Set[str] = set()
    for col, rep in representatives.items():
        if col == rep:
            keep.add(col)
        else:
            drop.add(col)
    # Ensure there is no overlap; if a col is both (shouldn't happen), prefer keep
    drop -= keep
    return sorted(keep), sorted(drop)


def _write_artifacts(output_dir: Path, kept: List[str], pairs: List[Tuple[str, str, float]], representatives: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # kept_features.yaml
    kept_path = output_dir / "kept_features.yaml"
    try:
        if yaml is not None:
            with open(kept_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(kept, f, sort_keys=False)
        else:
            # Fallback to a simple text list
            with open(kept_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(kept))
    except Exception:
        pass

    # dropped_pairs.csv
    if pairs:
        rows = []
        for a, b, v in pairs:
            rep = representatives.get(a, a)
            # Drop the non-representative of the pair
            drop = b if rep == a else a
            keep = rep
            rows.append({"col_a": a, "col_b": b, "corr": v, "kept": keep, "dropped": drop})
        df_pairs = pd.DataFrame(rows)
        df_pairs.to_csv(output_dir / "dropped_pairs.csv", index=False)

    # summary.md
    kept_count = len(kept)
    pair_count = len(pairs)
    total_cols = len({c for ab in pairs for c in ab[:2]})
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Correlation Filter Summary\n\n")
        f.write(f"Features kept: {kept_count}\n\n")
        f.write(f"Correlated pairs found: {pair_count}\n\n")
        f.write(f"Columns seen in pairs: {total_cols}\n\n")
        f.write("Selection policy: min missingness -> max variance -> lexicographic.\n")

    # Optional heatmap (best-effort)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        # Limit size for readability
        max_cols_for_plot = 75
        plot_cols = kept[:max_cols_for_plot]
        if len(plot_cols) >= 2:
            plt.figure(figsize=(min(18, 0.22 * len(plot_cols) + 4), min(12, 0.22 * len(plot_cols) + 4)))
            # Use a dummy DataFrame with zeros; heatmap will be overwritten by caller providing corr
            # Caller will pass corr separately; here we just document kept features
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / "corr_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Filter highly correlated features using training pipeline output.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Absolute correlation threshold for grouping (default: 0.95)")
    parser.add_argument("--output-dir", type=str, default="data/feature_selection", help="Directory to write artifacts")
    parser.add_argument("--exclude-cols", type=str, nargs="*", default=[], help="Additional columns to exclude from feature set")
    parser.add_argument("--seasons", type=str, default=None, help="Season range, e.g. '2020-2024' or '2023'")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row subsample size for faster correlation computation")

    args = parser.parse_args()

    # Import here to avoid heavy imports if just inspecting the module
    from kicktipp_predictor.data import DataLoader  # type: ignore

    loader = DataLoader()
    cfg = loader.config

    start_season, end_season = _parse_seasons_arg(args.seasons, cfg.paths.cache_dir, loader.get_current_season())

    all_matches = []
    for season in range(start_season, end_season + 1):
        all_matches.extend(loader.fetch_season_matches(season))

    if not all_matches:
        raise SystemExit("No matches found for selected seasons; cannot compute correlations.")

    df = loader.create_features_from_matches(all_matches)
    if df.empty:
        raise SystemExit("Feature DataFrame is empty; cannot compute correlations.")

    if args.max_samples is not None and args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)

    feature_cols = _detect_feature_columns(df, extra_exclude=args.exclude_cols)

    representatives, pairs = _build_correlation_groups(df, feature_cols, float(args.threshold))
    kept, dropped = _select_columns_by_groups(representatives)

    output_dir = Path(args.output_dir)
    _write_artifacts(output_dir, kept, pairs, representatives)

    # Also echo a short summary to stdout for quick inspection
    print(f"Seasons: {start_season}-{end_season}")
    print(f"Total numeric features considered: {len(feature_cols)}")
    print(f"Groups: {len(set(representatives.values()))}")
    print(f"Kept: {len(kept)}  Dropped: {len(dropped)}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()


