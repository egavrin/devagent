"""Coverage reporting and dashboard generation for DevAgent.

This module provides utilities to generate visual coverage reports
and track coverage trends over time.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageTrend:
    """Represents coverage trend data over time."""
    timestamp: str
    total_coverage: float
    branch_coverage: float
    files_covered: int
    total_files: int
    commit_hash: Optional[str] = None


class CoverageReporter:
    """Generate coverage reports and dashboards."""

    def __init__(self, project_root: Path = None):
        """Initialize coverage reporter.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root or Path.cwd()
        self.coverage_file = self.project_root / "coverage.json"
        self.trends_file = self.project_root / ".devagent" / "coverage_trends.json"
        self.trends_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self) -> Path:
        """Generate HTML coverage report.

        Returns:
            Path to generated HTML report
        """
        try:
            cmd = [
                sys.executable, "-m", "coverage", "html",
                "--directory=htmlcov",
                "--title=DevAgent Coverage Report"
            ]
            subprocess.run(cmd, cwd=self.project_root, check=True)

            report_path = self.project_root / "htmlcov" / "index.html"
            logger.info(f"HTML coverage report generated: {report_path}")
            return report_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise

    def generate_terminal_report(self, show_missing: bool = True) -> str:
        """Generate terminal coverage report.

        Args:
            show_missing: Include missing lines in report

        Returns:
            Report text
        """
        try:
            cmd = [sys.executable, "-m", "coverage", "report"]
            if show_missing:
                cmd.append("--show-missing")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate terminal report: {e}")
            return str(e)

    def get_coverage_summary(self) -> Dict:
        """Get coverage summary from coverage.json.

        Returns:
            Dictionary with coverage metrics
        """
        if not self.coverage_file.exists():
            return {
                "error": "No coverage data found. Run tests with coverage first.",
                "total_coverage": 0.0
            }

        with open(self.coverage_file, 'r') as f:
            data = json.load(f)

        totals = data.get("totals", {})

        return {
            "total_coverage": totals.get("percent_covered", 0.0),
            "branch_coverage": totals.get("percent_covered_display", "0.0"),
            "num_statements": totals.get("num_statements", 0),
            "covered_lines": totals.get("covered_lines", 0),
            "missing_lines": totals.get("missing_lines", 0),
            "excluded_lines": totals.get("excluded_lines", 0),
            "num_branches": totals.get("num_branches", 0),
            "covered_branches": totals.get("covered_branches", 0),
            "missing_branches": totals.get("missing_branches", 0),
            "files": len(data.get("files", {}))
        }

    def get_file_coverage(self, threshold: float = 95.0) -> Dict[str, List[Dict]]:
        """Get coverage by file, categorized by coverage level.

        Args:
            threshold: Coverage threshold for categorization

        Returns:
            Dictionary with categorized file coverage
        """
        if not self.coverage_file.exists():
            return {"error": "No coverage data available"}

        with open(self.coverage_file, 'r') as f:
            data = json.load(f)

        excellent = []  # >= threshold
        good = []       # 80-threshold
        fair = []       # 60-80
        poor = []       # < 60

        for file_path, file_data in data.get("files", {}).items():
            summary = file_data.get("summary", {})
            coverage = summary.get("percent_covered", 0.0)

            file_info = {
                "path": file_path,
                "coverage": coverage,
                "statements": summary.get("num_statements", 0),
                "missing": summary.get("missing_lines", 0)
            }

            if coverage >= threshold:
                excellent.append(file_info)
            elif coverage >= 80:
                good.append(file_info)
            elif coverage >= 60:
                fair.append(file_info)
            else:
                poor.append(file_info)

        return {
            "excellent": sorted(excellent, key=lambda x: x["coverage"], reverse=True),
            "good": sorted(good, key=lambda x: x["coverage"], reverse=True),
            "fair": sorted(fair, key=lambda x: x["coverage"], reverse=True),
            "poor": sorted(poor, key=lambda x: x["coverage"], reverse=True)
        }

    def save_coverage_trend(self) -> None:
        """Save current coverage as a trend data point."""
        summary = self.get_coverage_summary()

        if "error" in summary:
            logger.warning("Cannot save trend: no coverage data")
            return

        # Get current git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            commit_hash = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            commit_hash = None

        trend = CoverageTrend(
            timestamp=datetime.now().isoformat(),
            total_coverage=summary["total_coverage"],
            branch_coverage=float(summary["branch_coverage"].rstrip("%")),
            files_covered=summary["files"],
            total_files=summary["files"],
            commit_hash=commit_hash
        )

        # Load existing trends
        trends = self.load_coverage_trends()
        trends.append(trend)

        # Keep last 100 trends
        trends = trends[-100:]

        # Save updated trends
        with open(self.trends_file, 'w') as f:
            json.dump(
                [vars(t) for t in trends],
                f,
                indent=2
            )

        logger.info(f"Coverage trend saved: {trend.total_coverage:.1f}%")

    def load_coverage_trends(self) -> List[CoverageTrend]:
        """Load coverage trends from file.

        Returns:
            List of coverage trends
        """
        if not self.trends_file.exists():
            return []

        try:
            with open(self.trends_file, 'r') as f:
                data = json.load(f)
                return [CoverageTrend(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load trends: {e}")
            return []

    def get_coverage_trend_analysis(self) -> Dict:
        """Analyze coverage trends.

        Returns:
            Dictionary with trend analysis
        """
        trends = self.load_coverage_trends()

        if len(trends) < 2:
            return {
                "message": "Not enough data for trend analysis",
                "data_points": len(trends)
            }

        # Calculate trend direction
        recent = trends[-5:]  # Last 5 data points
        coverages = [t.total_coverage for t in recent]

        avg_recent = sum(coverages) / len(coverages)
        first_coverage = trends[0].total_coverage
        last_coverage = trends[-1].total_coverage

        trend_direction = "improving" if last_coverage > first_coverage else "declining"
        change = last_coverage - first_coverage

        return {
            "current_coverage": last_coverage,
            "initial_coverage": first_coverage,
            "change": change,
            "trend": trend_direction,
            "average_recent": avg_recent,
            "data_points": len(trends),
            "history": [
                {
                    "timestamp": t.timestamp,
                    "coverage": t.total_coverage,
                    "commit": t.commit_hash
                }
                for t in trends[-10:]  # Last 10 points
            ]
        }

    def generate_markdown_report(self) -> str:
        """Generate markdown coverage report.

        Returns:
            Markdown formatted report
        """
        summary = self.get_coverage_summary()
        file_coverage = self.get_file_coverage()
        trend_analysis = self.get_coverage_trend_analysis()

        report = ["# DevAgent Coverage Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overall summary
        report.append("## Overall Coverage\n")
        report.append(f"- **Total Coverage:** {summary['total_coverage']:.2f}%")
        report.append(f"- **Files Covered:** {summary['files']}")
        report.append(f"- **Total Statements:** {summary['num_statements']}")
        report.append(f"- **Covered Lines:** {summary['covered_lines']}")
        report.append(f"- **Missing Lines:** {summary['missing_lines']}")

        # Coverage status indicator
        total_cov = summary['total_coverage']
        if total_cov >= 95:
            status = "🟢 Excellent"
        elif total_cov >= 80:
            status = "🟡 Good"
        elif total_cov >= 60:
            status = "🟠 Fair"
        else:
            status = "🔴 Needs Improvement"

        report.append(f"\n**Status:** {status}\n")

        # Trend analysis
        if "current_coverage" in trend_analysis:
            report.append("## Coverage Trends\n")
            report.append(f"- **Current:** {trend_analysis['current_coverage']:.2f}%")
            report.append(f"- **Change:** {trend_analysis['change']:+.2f}%")
            report.append(f"- **Trend:** {trend_analysis['trend'].capitalize()}")
            report.append(f"- **Data Points:** {trend_analysis['data_points']}\n")

        # File coverage breakdown
        report.append("## Coverage by Category\n")

        if file_coverage.get("poor"):
            report.append(f"### 🔴 Needs Attention ({len(file_coverage['poor'])} files)\n")
            for file_info in file_coverage["poor"][:5]:
                report.append(f"- `{file_info['path']}`: {file_info['coverage']:.1f}%")
            report.append("")

        if file_coverage.get("fair"):
            report.append(f"### 🟠 Fair Coverage ({len(file_coverage['fair'])} files)\n")
            for file_info in file_coverage["fair"][:5]:
                report.append(f"- `{file_info['path']}`: {file_info['coverage']:.1f}%")
            report.append("")

        if file_coverage.get("good"):
            report.append(f"### 🟡 Good Coverage ({len(file_coverage['good'])} files)\n")
            report.append(f"{len(file_coverage['good'])} files with 80-95% coverage\n")

        if file_coverage.get("excellent"):
            report.append(f"### 🟢 Excellent Coverage ({len(file_coverage['excellent'])} files)\n")
            report.append(f"{len(file_coverage['excellent'])} files with ≥95% coverage\n")

        # Recommendations
        report.append("## Recommendations\n")
        if file_coverage.get("poor"):
            report.append("- Focus on improving coverage for files with < 60% coverage")
        if summary['total_coverage'] < 95:
            needed = 95 - summary['total_coverage']
            report.append(f"- Add tests to increase coverage by {needed:.1f}% to meet 95% threshold")
        else:
            report.append("- ✅ Coverage meets the 95% threshold!")

        return "\n".join(report)

    def generate_badge_data(self) -> Dict:
        """Generate badge data for README/docs.

        Returns:
            Badge configuration dict
        """
        summary = self.get_coverage_summary()
        coverage = summary['total_coverage']

        # Determine color based on coverage
        if coverage >= 95:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 60:
            color = "yellow"
        else:
            color = "red"

        return {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{coverage:.1f}%",
            "color": color
        }


def generate_report(format: str = "terminal", output_file: Optional[Path] = None) -> str:
    """Generate coverage report in specified format.

    Args:
        format: Report format (terminal, html, markdown, json)
        output_file: Output file path (optional)

    Returns:
        Report content or path to generated report
    """
    reporter = CoverageReporter()

    if format == "terminal":
        report = reporter.generate_terminal_report()
    elif format == "html":
        report_path = reporter.generate_html_report()
        return str(report_path)
    elif format == "markdown":
        report = reporter.generate_markdown_report()
    elif format == "json":
        import json
        summary = reporter.get_coverage_summary()
        report = json.dumps(summary, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Save to file if specified
    if output_file and format != "html":
        output_file.write_text(report)
        logger.info(f"Report saved to {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate coverage reports")
    parser.add_argument("--format", choices=["terminal", "html", "markdown", "json"],
                       default="terminal", help="Report format")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--save-trend", action="store_true",
                       help="Save current coverage as trend data")
    parser.add_argument("--show-trends", action="store_true",
                       help="Show coverage trend analysis")

    args = parser.parse_args()

    reporter = CoverageReporter()

    if args.save_trend:
        reporter.save_coverage_trend()
        print("✓ Coverage trend saved")

    if args.show_trends:
        trends = reporter.get_coverage_trend_analysis()
        print(json.dumps(trends, indent=2))

    # Generate report
    report = generate_report(format=args.format, output_file=args.output)
    if args.format != "html":
        print(report)