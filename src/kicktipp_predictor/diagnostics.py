"""Diagnostic tools for model calibration analysis.

This module provides diagnostic visualization tools to analyze model performance:
- Probability distribution histograms
- Calibration analysis
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .data import DataLoader
from .predictor import CascadedPredictor


def plot_draw_probability_distribution(output_dir: str | Path = "data/predictions") -> None:
    """Generate a histogram of draw probabilities to diagnose calibration issues.

    This diagnostic tool collects draw probabilities from all evaluated matches
    in the current season and visualizes their distribution. This helps diagnose
    whether the draw_model is producing a realistic range of probabilities or
    if it's overconfident (e.g., predicting P(Draw) > 0.5 for most matches).

    The histogram is saved as 'draw_probability_histogram.png' in the output directory.

    Args:
        output_dir: Directory to save the histogram image. Defaults to 'data/predictions'.
    """
    console = Console()

    console.print(
        Panel(
            "[bold cyan]DRAW PROBABILITY DISTRIBUTION DIAGNOSTIC[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    # Load models
    predictor = CascadedPredictor()
    try:
        predictor.load_models()
        console.print("[green]✓ Models loaded successfully[/green]")
    except FileNotFoundError:
        console.print("[red]No trained models found. Run training first.[/red]")
        raise SystemExit(1)

    # Load data
    data_loader = DataLoader()
    current_season = data_loader.get_current_season()
    console.print(f"Fetching matches for season [bold]{current_season}[/bold]...")

    season_matches = data_loader.fetch_season_matches(current_season)
    finished_matches = [m for m in season_matches if m.get("is_finished")]

    if not finished_matches:
        console.print("[yellow]No finished matches found.[/yellow]")
        return

    console.print(f"[green]✓ Found {len(finished_matches)} finished matches[/green]")

    # Get historical context for feature creation
    historical_matches = data_loader.fetch_historical_seasons(
        current_season - 5, current_season - 1
    )

    # Create features
    console.print("[cyan]Creating features...[/cyan]")
    features_df = data_loader.create_prediction_features(
        finished_matches, historical_matches
    )

    if features_df is None or len(features_df) == 0:
        console.print("[yellow]Insufficient features; skipping.[/yellow]")
        return

    # Make predictions
    console.print("[cyan]Generating predictions...[/cyan]")
    predictions = predictor.predict(features_df)

    # Extract draw probabilities
    draw_probs = [p.get("draw_probability", 0.0) for p in predictions]
    draw_probs = np.array(draw_probs, dtype=float)

    # Statistics
    mean_draw_prob = float(np.mean(draw_probs))
    median_draw_prob = float(np.median(draw_probs))
    std_draw_prob = float(np.std(draw_probs))
    min_draw_prob = float(np.min(draw_probs))
    max_draw_prob = float(np.max(draw_probs))

    # Count matches with high draw probability
    n_high_prob = int(np.sum(draw_probs > 0.5))
    n_very_high_prob = int(np.sum(draw_probs > 0.7))

    # Display statistics
    stats_table = Table(title="Draw Probability Statistics", box=box.SIMPLE_HEAVY)
    stats_table.add_column("Metric", justify="left")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Mean", f"{mean_draw_prob:.4f}")
    stats_table.add_row("Median", f"{median_draw_prob:.4f}")
    stats_table.add_row("Std Dev", f"{std_draw_prob:.4f}")
    stats_table.add_row("Min", f"{min_draw_prob:.4f}")
    stats_table.add_row("Max", f"{max_draw_prob:.4f}")
    stats_table.add_row("Matches with P(Draw) > 0.5", f"{n_high_prob} ({n_high_prob/len(draw_probs):.1%})")
    stats_table.add_row("Matches with P(Draw) > 0.7", f"{n_very_high_prob} ({n_very_high_prob/len(draw_probs):.1%})")

    console.print(stats_table)

    # Calculate actual draw rate
    actual_labels = []
    for pred, match in zip(predictions, finished_matches):
        actual_home = int(match.get("home_score", 0))
        actual_away = int(match.get("away_score", 0))
        if actual_home > actual_away:
            actual_labels.append("H")
        elif actual_away > actual_home:
            actual_labels.append("A")
        else:
            actual_labels.append("D")

    actual_draw_rate = float(actual_labels.count("D") / len(actual_labels))

    # Comparison table
    comparison_table = Table(title="Predicted vs Actual Draw Rate", box=box.SIMPLE_HEAVY)
    comparison_table.add_column("Metric", justify="left")
    comparison_table.add_column("Value", justify="right")
    comparison_table.add_row("Average predicted P(Draw)", f"{mean_draw_prob:.1%}")
    comparison_table.add_row("Actual draw rate", f"{actual_draw_rate:.1%}")
    comparison_table.add_row("Difference", f"{mean_draw_prob - actual_draw_rate:+.1%}")
    
    diff_status = "OK" if abs(mean_draw_prob - actual_draw_rate) < 0.10 else "ISSUE"
    comparison_table.add_row("Status", diff_status)

    console.print("\n")
    console.print(comparison_table)

    # Generate histogram
    try:
        import matplotlib.pyplot as plt

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n_bins = 50
        counts, bins, patches = ax.hist(
            draw_probs, bins=n_bins, edgecolor="black", alpha=0.7, color="#f39c12"
        )

        # Highlight high-probability regions
        for i, patch in enumerate(patches):
            if bins[i] >= 0.5:
                patch.set_facecolor("#e74c3c")
                patch.set_alpha(0.8)

        # Add vertical lines for mean and actual draw rate
        ax.axvline(
            mean_draw_prob, color="blue", linestyle="--", linewidth=2, label=f"Mean P(Draw) = {mean_draw_prob:.2%}"
        )
        ax.axvline(
            actual_draw_rate, color="green", linestyle="--", linewidth=2, label=f"Actual Draw Rate = {actual_draw_rate:.2%}"
        )

        ax.set_xlabel("Draw Probability", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Matches", fontsize=12, fontweight="bold")
        ax.set_title("Distribution of Predicted Draw Probabilities", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Save histogram
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hist_path = output_dir / "draw_probability_histogram.png"
        
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()

        console.print(f"\n[green]✓ Histogram saved to {hist_path}[/green]")

    except ImportError as e:
        console.print(f"[yellow]Warning: Could not import matplotlib: {e}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Warning: Error generating histogram: {e}[/yellow]")

    # Save raw data for further analysis
    try:
        output_dir = Path(output_dir)
        data_path = output_dir / "draw_probability_data.csv"
        
        # Create DataFrame with predictions and actual results
        rows = []
        for idx, (pred, match) in enumerate(zip(predictions, finished_matches)):
            row = {
                "match_id": pred.get("match_id"),
                "matchday": pred.get("matchday"),
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "draw_probability": pred.get("draw_probability"),
                "home_win_probability": pred.get("home_win_probability"),
                "away_win_probability": pred.get("away_win_probability"),
                "actual_home_score": match.get("home_score"),
                "actual_away_score": match.get("away_score"),
                "actual_result": actual_labels[idx],
            }
            rows.append(row)
        
        pd.DataFrame(rows).to_csv(data_path, index=False)
        console.print(f"[green]✓ Data saved to {data_path}[/green]")

    except Exception as e:
        console.print(f"[yellow]Warning: Could not save data: {e}[/yellow]")

    console.print("\n[bold]Diagnostic complete![/bold]")
