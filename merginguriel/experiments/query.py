"""
Query CLI for experiment results.

Provides a simple interface to query the experiment database.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from merginguriel.experiments.db import ExperimentDB


def format_table(rows: list, headers: list) -> str:
    """Format rows as a text table."""
    if not rows:
        return "No results found."

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val) if val is not None else ""))

    # Build table
    lines = []

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = " | ".join(
            str(val if val is not None else "").ljust(widths[i])
            for i, val in enumerate(row)
        )
        lines.append(row_line)

    return "\n".join(lines)


def cmd_list(db: ExperimentDB, args) -> None:
    """List experiments with optional filters."""
    records = db.find(
        locale=args.locale,
        method=args.method,
        similarity_type=args.similarity,
        status=args.status,
        ablation_name=args.ablation,
        limit=args.limit,
    )

    if not records:
        print("No experiments found.")
        return

    headers = ["ID", "Locale", "Method", "Similarity", "NumLang", "Accuracy", "Status"]
    rows = [
        (r.id, r.locale, r.method, r.similarity_type, r.num_languages,
         f"{r.accuracy:.4f}" if r.accuracy else "-", r.status)
        for r in records
    ]

    print(format_table(rows, headers))
    print(f"\nTotal: {len(records)} experiments")


def cmd_summary(db: ExperimentDB, args) -> None:
    """Show summary statistics."""
    print("=== Summary by Locale ===\n")
    by_locale = db.summary_by_locale()
    if by_locale:
        headers = ["Locale", "Total", "Completed", "Avg Acc", "Best Acc"]
        rows = [
            (d["locale"], d["total"], d["completed"],
             f"{d['avg_accuracy']:.4f}" if d["avg_accuracy"] else "-",
             f"{d['best_accuracy']:.4f}" if d["best_accuracy"] else "-")
            for d in by_locale
        ]
        print(format_table(rows, headers))
    else:
        print("No data.")

    print("\n=== Summary by Method ===\n")
    by_method = db.summary_by_method()
    if by_method:
        headers = ["Method", "Total", "Completed", "Avg Acc"]
        rows = [
            (d["method"], d["total"], d["completed"],
             f"{d['avg_accuracy']:.4f}" if d["avg_accuracy"] else "-")
            for d in by_method
        ]
        print(format_table(rows, headers))
    else:
        print("No data.")


def cmd_best(db: ExperimentDB, args) -> None:
    """Show best configuration per locale."""
    best = db.best_config_per_locale()

    if not best:
        print("No completed experiments found.")
        return

    headers = ["Locale", "Method", "Similarity", "NumLang", "Accuracy"]
    rows = [
        (d["locale"], d["method"], d["similarity_type"], d["num_languages"],
         f"{d['accuracy']:.4f}" if d["accuracy"] else "-")
        for d in best
    ]

    print("=== Best Configuration per Locale ===\n")
    print(format_table(rows, headers))


def cmd_ablation(db: ExperimentDB, args) -> None:
    """Show results for a specific ablation study."""
    results = db.ablation_comparison(args.name)

    if not results:
        print(f"No results found for ablation '{args.name}'.")
        return

    headers = ["Locale", "Method", "Similarity", "NumLang", "IncTar", "Accuracy", "Status"]
    rows = [
        (d["locale"], d["method"], d["similarity_type"], d["num_languages"],
         "Yes" if d["include_target"] else "No",
         f"{d['accuracy']:.4f}" if d["accuracy"] else "-", d["status"])
        for d in results
    ]

    print(f"=== Ablation: {args.name} ===\n")
    print(format_table(rows, headers))


def cmd_stats(db: ExperimentDB, args) -> None:
    """Show database statistics."""
    stats = db.stats()

    print("=== Database Statistics ===\n")
    print(f"Database: {stats['db_path']}")
    print(f"Total experiments: {stats['total']}")
    print("\nBy status:")
    for status, count in stats["by_status"].items():
        print(f"  {status}: {count}")


def cmd_export(db: ExperimentDB, args) -> None:
    """Export experiments to CSV."""
    output = args.output or "experiments_export.csv"
    db.export_csv(output)
    print(f"Exported to {output}")


def cmd_sql(db: ExperimentDB, args) -> None:
    """Execute raw SQL query."""
    try:
        rows = db.query(args.query)
        if rows:
            headers = rows[0].keys()
            data = [tuple(row) for row in rows]
            print(format_table(data, list(headers)))
            print(f"\n{len(rows)} rows returned")
        else:
            print("Query returned no results.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query experiment results database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                          # List all experiments
  %(prog)s list --locale sq-AL           # Filter by locale
  %(prog)s list --status completed       # Filter by status
  %(prog)s summary                       # Show summary statistics
  %(prog)s best                          # Show best config per locale
  %(prog)s ablation similarity_ablation  # Show ablation results
  %(prog)s sql "SELECT * FROM experiments WHERE accuracy > 0.6"
        """
    )
    parser.add_argument("--db", type=str, default="experiments.db",
                        help="Path to database (default: experiments.db)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--locale", help="Filter by locale")
    list_parser.add_argument("--method", help="Filter by method")
    list_parser.add_argument("--similarity", help="Filter by similarity type")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--ablation", help="Filter by ablation name")
    list_parser.add_argument("--limit", type=int, default=50, help="Limit results")

    # summary command
    subparsers.add_parser("summary", help="Show summary statistics")

    # best command
    subparsers.add_parser("best", help="Show best config per locale")

    # ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Show ablation results")
    ablation_parser.add_argument("name", help="Ablation name")

    # stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # export command
    export_parser = subparsers.add_parser("export", help="Export to CSV")
    export_parser.add_argument("--output", "-o", help="Output file path")

    # sql command
    sql_parser = subparsers.add_parser("sql", help="Execute raw SQL")
    sql_parser.add_argument("query", help="SQL query to execute")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize database
    db = ExperimentDB(args.db)

    # Dispatch to command handler
    commands = {
        "list": cmd_list,
        "summary": cmd_summary,
        "best": cmd_best,
        "ablation": cmd_ablation,
        "stats": cmd_stats,
        "export": cmd_export,
        "sql": cmd_sql,
    }

    commands[args.command](db, args)


if __name__ == "__main__":
    main()
