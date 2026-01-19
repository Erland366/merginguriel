"""
Plotting utilities for ablation experiment results.

Generates comparison plots and tables from the experiment database.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from merginguriel.experiments.db import ExperimentDB


class AblationPlotter:
    """Generate plots from ablation experiment results."""

    def __init__(self, db: ExperimentDB):
        self.db = db

    def _get_completed_experiments(self, ablation_name: Optional[str] = None) -> pd.DataFrame:
        """Get completed experiments as a DataFrame."""
        if ablation_name:
            records = self.db.find(ablation_name=ablation_name, status="completed")
        else:
            records = self.db.find(status="completed")

        if not records:
            return pd.DataFrame()

        data = [r.to_dict() for r in records]
        return pd.DataFrame(data)

    def plot_ablation_comparison(
        self,
        ablation_name: str,
        sweep_param: str,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> Optional[plt.Figure]:
        """
        Create a bar plot comparing results across the swept parameter.

        Args:
            ablation_name: Name of the ablation study
            sweep_param: The parameter that was swept (e.g., 'similarity_type', 'num_languages')
            output_path: Path to save the figure (optional)
            figsize: Figure size

        Returns:
            matplotlib Figure or None if no data
        """
        df = self._get_completed_experiments(ablation_name)
        if df.empty:
            print(f"No completed experiments found for ablation '{ablation_name}'")
            return None

        # Pivot: locales as x-axis, sweep_param values as groups
        pivot = df.pivot_table(
            index='locale',
            columns=sweep_param,
            values='accuracy',
            aggfunc='mean'
        )

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        pivot.plot(kind='bar', ax=ax, width=0.8)

        ax.set_xlabel('Locale')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Ablation: {ablation_name}\nComparing {sweep_param}')
        ax.legend(title=sweep_param)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        return fig

    def plot_scaling(
        self,
        ablation_name: str,
        x_param: str = 'num_languages',
        output_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> Optional[plt.Figure]:
        """
        Create a line plot showing scaling behavior.

        Args:
            ablation_name: Name of the ablation study
            x_param: Parameter for x-axis (e.g., 'num_languages')
            output_path: Path to save the figure
            figsize: Figure size

        Returns:
            matplotlib Figure or None if no data
        """
        df = self._get_completed_experiments(ablation_name)
        if df.empty:
            print(f"No completed experiments found for ablation '{ablation_name}'")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each locale as a line
        for locale in df['locale'].unique():
            locale_df = df[df['locale'] == locale].sort_values(x_param)
            ax.plot(locale_df[x_param], locale_df['accuracy'],
                   marker='o', label=locale, linewidth=2, markersize=8)

        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Scaling: {ablation_name}')
        ax.legend(title='Locale')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        return fig

    def plot_method_heatmap(
        self,
        ablation_name: Optional[str] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> Optional[plt.Figure]:
        """
        Create a heatmap of accuracy by locale and method.

        Args:
            ablation_name: Filter by ablation name (optional)
            output_path: Path to save the figure
            figsize: Figure size

        Returns:
            matplotlib Figure or None if no data
        """
        df = self._get_completed_experiments(ablation_name)
        if df.empty:
            print("No completed experiments found")
            return None

        # Pivot for heatmap
        pivot = df.pivot_table(
            index='locale',
            columns='method',
            values='accuracy',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy')

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           color=text_color, fontsize=9)

        ax.set_title('Accuracy by Locale and Method')
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        return fig

    def generate_latex_table(
        self,
        ablation_name: str,
        sweep_param: str,
    ) -> str:
        """
        Generate a LaTeX table for the ablation results.

        Args:
            ablation_name: Name of the ablation study
            sweep_param: The parameter that was swept

        Returns:
            LaTeX table string
        """
        df = self._get_completed_experiments(ablation_name)
        if df.empty:
            return "% No data available"

        pivot = df.pivot_table(
            index='locale',
            columns=sweep_param,
            values='accuracy',
            aggfunc='mean'
        )

        # Find best value per row for bolding
        best_per_row = pivot.idxmax(axis=1)

        # Build LaTeX
        cols = pivot.columns.tolist()
        header = " & ".join(["Locale"] + [str(c) for c in cols]) + " \\\\"

        rows = []
        for locale in pivot.index:
            row_vals = []
            for col in cols:
                val = pivot.loc[locale, col]
                if pd.isna(val):
                    row_vals.append("-")
                elif best_per_row[locale] == col:
                    row_vals.append(f"\\textbf{{{val:.4f}}}")
                else:
                    row_vals.append(f"{val:.4f}")
            rows.append(f"{locale} & " + " & ".join(row_vals) + " \\\\")

        # Calculate averages
        avg_row = []
        for col in cols:
            avg = pivot[col].mean()
            avg_row.append(f"{avg:.4f}")
        rows.append("\\midrule")
        rows.append("Average & " + " & ".join(avg_row) + " \\\\")

        latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Ablation: {ablation_name} - Comparing {sweep_param}}}
\\begin{{tabular}}{{l{'c' * len(cols)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

        return latex

    def generate_markdown_report(
        self,
        ablation_name: str,
        sweep_param: str,
    ) -> str:
        """
        Generate a Markdown report for the ablation results.

        Args:
            ablation_name: Name of the ablation study
            sweep_param: The parameter that was swept

        Returns:
            Markdown report string
        """
        df = self._get_completed_experiments(ablation_name)
        if df.empty:
            return f"# Ablation: {ablation_name}\n\nNo completed experiments found."

        pivot = df.pivot_table(
            index='locale',
            columns=sweep_param,
            values='accuracy',
            aggfunc='mean'
        )

        # Build markdown table
        cols = pivot.columns.tolist()
        header = "| Locale | " + " | ".join(str(c) for c in cols) + " |"
        separator = "|--------|" + "|".join(["--------"] * len(cols)) + "|"

        rows = []
        for locale in pivot.index:
            row_vals = []
            best_val = pivot.loc[locale].max()
            for col in cols:
                val = pivot.loc[locale, col]
                if pd.isna(val):
                    row_vals.append("-")
                elif val == best_val:
                    row_vals.append(f"**{val:.4f}**")
                else:
                    row_vals.append(f"{val:.4f}")
            rows.append(f"| {locale} | " + " | ".join(row_vals) + " |")

        # Averages
        avg_row = []
        for col in cols:
            avg_row.append(f"{pivot[col].mean():.4f}")
        rows.append(f"| **Average** | " + " | ".join(avg_row) + " |")

        # Summary stats
        best_overall = pivot.mean().idxmax()
        best_avg = pivot.mean().max()

        report = f"""# Ablation: {ablation_name}

## Summary

- **Sweep Parameter:** {sweep_param}
- **Values Tested:** {', '.join(str(c) for c in cols)}
- **Locales:** {', '.join(pivot.index.tolist())}
- **Best Overall:** {best_overall} (avg accuracy: {best_avg:.4f})

## Results

{header}
{separator}
{chr(10).join(rows)}

## Key Findings

"""
        # Add per-locale winners
        for locale in pivot.index:
            best = pivot.loc[locale].idxmax()
            val = pivot.loc[locale, best]
            report += f"- **{locale}**: Best with {sweep_param}={best} ({val:.4f})\n"

        return report


def main():
    """CLI entry point for ablation plotting."""
    parser = argparse.ArgumentParser(description="Plot ablation experiment results")
    parser.add_argument("--db", type=str, default="experiments.db",
                        help="Path to database")
    parser.add_argument("--output-dir", "-o", type=str, default="plots",
                        help="Output directory for plots")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # comparison command
    comp_parser = subparsers.add_parser("comparison", help="Bar plot comparing sweep values")
    comp_parser.add_argument("ablation", help="Ablation name")
    comp_parser.add_argument("sweep_param", help="Swept parameter name")

    # scaling command
    scale_parser = subparsers.add_parser("scaling", help="Line plot showing scaling")
    scale_parser.add_argument("ablation", help="Ablation name")
    scale_parser.add_argument("--x-param", default="num_languages", help="X-axis parameter")

    # heatmap command
    heat_parser = subparsers.add_parser("heatmap", help="Heatmap of accuracy")
    heat_parser.add_argument("--ablation", help="Filter by ablation name")

    # latex command
    latex_parser = subparsers.add_parser("latex", help="Generate LaTeX table")
    latex_parser.add_argument("ablation", help="Ablation name")
    latex_parser.add_argument("sweep_param", help="Swept parameter name")

    # markdown command
    md_parser = subparsers.add_parser("markdown", help="Generate Markdown report")
    md_parser.add_argument("ablation", help="Ablation name")
    md_parser.add_argument("sweep_param", help="Swept parameter name")

    # all command - generate all plots for an ablation
    all_parser = subparsers.add_parser("all", help="Generate all plots and reports")
    all_parser.add_argument("ablation", help="Ablation name")
    all_parser.add_argument("sweep_param", help="Swept parameter name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup
    db = ExperimentDB(args.db)
    plotter = AblationPlotter(db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.command == "comparison":
        output_path = output_dir / f"{args.ablation}_comparison.png"
        plotter.plot_ablation_comparison(args.ablation, args.sweep_param, str(output_path))
        plt.show()

    elif args.command == "scaling":
        output_path = output_dir / f"{args.ablation}_scaling.png"
        plotter.plot_scaling(args.ablation, args.x_param, str(output_path))
        plt.show()

    elif args.command == "heatmap":
        name = args.ablation or "all"
        output_path = output_dir / f"{name}_heatmap.png"
        plotter.plot_method_heatmap(args.ablation, str(output_path))
        plt.show()

    elif args.command == "latex":
        latex = plotter.generate_latex_table(args.ablation, args.sweep_param)
        print(latex)
        # Also save to file
        output_path = output_dir / f"{args.ablation}_table.tex"
        output_path.write_text(latex)
        print(f"\nSaved to {output_path}")

    elif args.command == "markdown":
        md = plotter.generate_markdown_report(args.ablation, args.sweep_param)
        print(md)
        # Also save to file
        output_path = output_dir / f"{args.ablation}_report.md"
        output_path.write_text(md)
        print(f"\nSaved to {output_path}")

    elif args.command == "all":
        print(f"Generating all outputs for {args.ablation}...")

        # Comparison plot
        plotter.plot_ablation_comparison(
            args.ablation, args.sweep_param,
            str(output_dir / f"{args.ablation}_comparison.png")
        )

        # Heatmap
        plotter.plot_method_heatmap(
            args.ablation,
            str(output_dir / f"{args.ablation}_heatmap.png")
        )

        # LaTeX table
        latex = plotter.generate_latex_table(args.ablation, args.sweep_param)
        (output_dir / f"{args.ablation}_table.tex").write_text(latex)
        print(f"Saved LaTeX to {output_dir / f'{args.ablation}_table.tex'}")

        # Markdown report
        md = plotter.generate_markdown_report(args.ablation, args.sweep_param)
        (output_dir / f"{args.ablation}_report.md").write_text(md)
        print(f"Saved Markdown to {output_dir / f'{args.ablation}_report.md'}")

        print("\nAll outputs generated!")


if __name__ == "__main__":
    main()
