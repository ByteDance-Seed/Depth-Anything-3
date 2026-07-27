# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Beautiful metrics printing utilities for benchmark evaluation.

Provides colorized, well-formatted tabular output for evaluation results.
Supports highlighting best/worst values and grouping by dataset/mode.
"""

import argparse
import json
import os
import re
from typing import Dict as TDict, List, Optional

import numpy as np


# ANSI color codes for terminal output
class Colors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bold variants
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"

    # Background
    BG_DARK = "\033[48;5;236m"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from string for length calculation."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def colorize_value(
    value: str,
    is_best: bool = False,
    is_worst: bool = False,
    lower_is_better: bool = False,
) -> str:
    """
    Apply color to a metric value based on whether it's best/worst.

    Args:
        value: String representation of the value
        is_best: Whether this is the best value in its column
        is_worst: Whether this is the worst value in its column
        lower_is_better: If True, lower values are better (e.g., error metrics)

    Returns:
        Colorized string
    """
    if lower_is_better:
        # For metrics like error/distance, lower is better
        if is_best:
            return f"{Colors.BOLD_GREEN}{value}{Colors.RESET}"
        elif is_worst:
            return f"{Colors.BOLD_RED}{value}{Colors.RESET}"
    else:
        # For metrics like accuracy/AUC, higher is better
        if is_best:
            return f"{Colors.BOLD_GREEN}{value}{Colors.RESET}"
        elif is_worst:
            return f"{Colors.BOLD_RED}{value}{Colors.RESET}"
    return value


class MetricsPrinter:
    """
    Beautiful tabular metrics printer with color support.

    Features:
    - Colorized best/worst values
    - Grouped by dataset and evaluation mode
    - Automatic column width calculation
    - Support for multiple input directories comparison
    """

    # Metrics where lower values are better
    # Add "rel" so abs_rel / points_rel / depth_rel etc. are colorized correctly.
    LOWER_IS_BETTER = {"comp", "acc", "overall", "error", "loss", "rmse", "mae", "rel"}

    def __init__(self, use_color: bool = True):
        """
        Initialize the printer.

        Args:
            use_color: Whether to use ANSI colors in output
        """
        self.use_color = use_color

    def print_results(self, metrics: TDict[str, dict], summary_only: bool = True) -> None:
        """
        Print evaluation metrics in a beautiful tabular format.

        Args:
            metrics: Dictionary mapping "dataset_mode" to metric results
            summary_only: If True, only print summary table. If False, print per-dataset details too.
        """
        if not metrics:
            print(f"\n{Colors.BOLD_RED}❌ No evaluation metrics found{Colors.RESET}")
            return

        if not summary_only:
            self._print_header()
            grouped = self._group_by_dataset(metrics)

            for dataset, modes_data in grouped.items():
                self._print_dataset_section(dataset, modes_data)

        # Print summary table with average metrics across datasets
        self._print_summary(metrics)

        self._print_footer()

    def print_comparison(
        self,
        metrics_list: List[TDict[str, dict]],
        labels: List[str],
    ) -> None:
        """
        Print comparison table for multiple evaluation runs.

        Args:
            metrics_list: List of metrics dictionaries
            labels: Labels for each metrics dictionary
        """
        if not metrics_list or not all(metrics_list):
            print(f"\n{Colors.BOLD_RED}❌ No metrics to compare{Colors.RESET}")
            return

        # Collect all datasets and modes
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        self._print_header("COMPARISON")

        for key in sorted(all_keys):
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                dataset, mode = parts[0], parts[1]
            else:
                dataset, mode = key, "unknown"

            print(f"\n{Colors.BOLD_CYAN}📊 {dataset.upper()} - {mode.upper()}{Colors.RESET}")
            print("-" * 100)

            # Collect metrics from all runs
            all_metric_names = set()
            for metrics in metrics_list:
                if key in metrics and "mean" in metrics[key]:
                    all_metric_names.update(metrics[key]["mean"].keys())

            if not all_metric_names:
                continue

            # Build comparison table
            metric_width = max(15, max(len(m) for m in all_metric_names) + 2)
            label_width = max(15, max(len(l) for l in labels) + 2)

            # Header
            header = f"{'Metric':<{metric_width}}"
            for label in labels:
                header += f"{label:<{label_width}}"
            print(header)
            print("-" * len(strip_ansi(header)))

            # Collect values for highlighting
            for metric_name in sorted(all_metric_names):
                values = []
                for metrics in metrics_list:
                    if key in metrics and "mean" in metrics[key]:
                        val = metrics[key]["mean"].get(metric_name)
                        values.append(val if val is not None else float("nan"))
                    else:
                        values.append(float("nan"))

                # Find best/worst
                valid_values = [v for v in values if not (v != v)]  # Filter NaN
                if valid_values:
                    lower_better = any(
                        lb in metric_name.lower() for lb in self.LOWER_IS_BETTER
                    )
                    best_val = min(valid_values) if lower_better else max(valid_values)
                    worst_val = max(valid_values) if lower_better else min(valid_values)
                else:
                    best_val = worst_val = None

                # Print row
                row = f"{metric_name:<{metric_width}}"
                for val in values:
                    if val != val:  # NaN check
                        val_str = "N/A"
                    else:
                        val_str = f"{val:.4f}"
                        if self.use_color and len(valid_values) > 1:
                            lower_better = any(
                                lb in metric_name.lower() for lb in self.LOWER_IS_BETTER
                            )
                            is_best = abs(val - best_val) < 1e-8 if best_val else False
                            is_worst = abs(val - worst_val) < 1e-8 if worst_val else False
                            val_str_padded = f"{val_str:<{label_width}}"
                            val_str = colorize_value(
                                val_str_padded, is_best, is_worst, lower_better
                            )
                            row += val_str
                            continue
                    row += f"{val_str:<{label_width}}"
                print(row)

        self._print_footer()

    def _print_header(self, title: str = "EVALUATION RESULTS") -> None:
        """Print report header."""
        width = 100
        print()
        print("=" * width)
        print(f"{Colors.BOLD_CYAN}📊 DEPTH ANYTHING 3 {title}{Colors.RESET}")
        print("=" * width)

    def _print_footer(self) -> None:
        """Print report footer."""
        width = 100
        print()
        print("=" * width)
        print(f"{Colors.BOLD_GREEN}✅ Evaluation Complete{Colors.RESET}")
        print("=" * width)
        print()

    def _group_by_dataset(self, metrics: TDict[str, dict]) -> TDict[str, dict]:
        """Group metrics by dataset."""
        grouped = {}
        for key, data in metrics.items():
            if not isinstance(data, dict) or "mean" not in data:
                continue
            # Parse key format: "dataset_mode" (e.g., "dtu_recon_unposed")
            parts = key.split("_", 1)
            if len(parts) == 2:
                dataset, mode = parts
                if dataset not in grouped:
                    grouped[dataset] = {}
                grouped[dataset][mode] = data
        return grouped

    def _print_dataset_section(self, dataset: str, modes_data: TDict[str, dict]) -> None:
        """Print metrics section for a single dataset."""
        print(f"\n{Colors.BOLD_MAGENTA}🔍 {dataset.upper()}{Colors.RESET}")
        print("-" * 100)

        # Collect all unique metrics across all modes
        all_metrics = set()
        for mode_data in modes_data.values():
            all_metrics.update(mode_data["mean"].keys())
        all_metrics = sorted(list(all_metrics))

        if not all_metrics:
            print("  No metrics available")
            return

        # Calculate column widths
        metric_width = max(18, max(len(m) for m in all_metrics) + 2)
        mode_width = 18
        modes = list(modes_data.keys())

        # Print header
        header = f"{'Metric':<{metric_width}}"
        for mode in modes:
            header += f"{mode.upper():<{mode_width}}"
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print("-" * len(header))

        # Print each metric row
        for metric in all_metrics:
            row = f"{metric:<{metric_width}}"

            # Collect values for this metric across modes
            values = []
            for mode in modes:
                if metric in modes_data[mode]["mean"]:
                    values.append(modes_data[mode]["mean"][metric])
                else:
                    values.append(None)

            # Find best/worst values
            valid_values = [v for v in values if v is not None]
            if valid_values:
                lower_better = any(lb in metric.lower() for lb in self.LOWER_IS_BETTER)
                best_val = min(valid_values) if lower_better else max(valid_values)
                worst_val = max(valid_values) if lower_better else min(valid_values)
            else:
                best_val = worst_val = None

            # Format each value
            for val in values:
                if val is None:
                    row += f"{'N/A':<{mode_width}}"
                else:
                    val_str = f"{val:.4f}"
                    if self.use_color and len(valid_values) > 1:
                        is_best = abs(val - best_val) < 1e-8 if best_val else False
                        is_worst = abs(val - worst_val) < 1e-8 if worst_val else False
                        lower_better = any(
                            lb in metric.lower() for lb in self.LOWER_IS_BETTER
                        )
                        # Pad before colorizing to maintain alignment
                        val_str_padded = f"{val_str:<{mode_width}}"
                        row += colorize_value(
                            val_str_padded, is_best, is_worst, lower_better
                        )
                    else:
                        row += f"{val_str:<{mode_width}}"
            print(row)

        # Show scene counts
        scene_info = []
        for mode, mode_data in modes_data.items():
            scene_count = len([k for k in mode_data.keys() if k != "mean"])
            scene_info.append(f"{mode}: {scene_count} scenes")
        print(f"\n{Colors.CYAN}📈 {' | '.join(scene_info)}{Colors.RESET}")

    def _print_summary(self, metrics: TDict[str, dict]) -> None:
        """
        Print summary table with key metrics across all datasets.

        Format: One row per metric, datasets as columns.
        Order: HiRoom, ETH3D, DTU, 7Scenes, ScanNet++, (DTU-64 for pose only)
        """
        print(f"\n{Colors.BOLD_CYAN}{'=' * 120}{Colors.RESET}")
        print(f"{Colors.BOLD_CYAN}📊 SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD_CYAN}{'=' * 120}{Colors.RESET}")

        # Dataset display order and names
        DATASET_ORDER = ["hiroom", "eth3d", "dtu", "7scenes", "scannetpp", "dtu64"]
        DATASET_DISPLAY = {
            "hiroom": "HiRoom",
            "eth3d": "ETH3D",
            "dtu": "DTU",
            "7scenes": "7Scenes",
            "scannetpp": "ScanNet++",
            "dtu64": "DTU-64",
        }

        # Collect all metrics into a structured dict
        # metric_data[dataset][mode] = {"Auc_3": x, "Auc_30": x, "fscore": x, "overall": x}
        metric_data = {}
        for key, data in metrics.items():
            if not isinstance(data, dict) or "mean" not in data:
                continue
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            dataset, mode = parts
            dataset_lower = dataset.lower()
            if dataset_lower not in metric_data:
                metric_data[dataset_lower] = {}
            metric_data[dataset_lower][mode] = data["mean"]

        col_width = 12

        def fmt_val(val):
            """Format value or return N/A."""
            if val is None:
                return "N/A"
            return f"{val:.4f}"

        def get_metric(dataset, mode, metric_name):
            """Get metric value or None."""
            if dataset not in metric_data:
                return None
            if mode not in metric_data[dataset]:
                return None
            return metric_data[dataset][mode].get(metric_name)

        # ============ POSE METRICS ============
        print(f"\n{Colors.BOLD_MAGENTA}🎯 POSE ESTIMATION{Colors.RESET}")
        
        # Pose: show all datasets except DTU (keep DTU-64 only)
        # Order: HiRoom, ETH3D, DTU-64, 7Scenes, ScanNet++
        pose_datasets = ["hiroom", "eth3d", "dtu64", "7scenes", "scannetpp"]
        
        # Header: Avg first, then datasets
        header = f"{'Metric':<15}{'Avg':<{col_width}}"
        for ds in pose_datasets:
            header += f"{DATASET_DISPLAY[ds]:<{col_width}}"
        print("-" * len(strip_ansi(header)))
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print("-" * len(strip_ansi(header)))

        # Helper to get metric with fallback names
        def get_pose_metric(dataset, metric_name):
            """Get pose metric with fallback for different naming conventions."""
            # Try different naming conventions
            names = {
                "Auc3": ["Auc_3", "auc03", "auc_3", "AUC_3", "Auc3", "auc3"],
                "Auc30": ["Auc_30", "auc30", "auc_30", "AUC_30", "Auc30"],
            }
            for name in names.get(metric_name, [metric_name]):
                val = get_metric(dataset, "pose", name)
                if val is not None:
                    return val
            return None

        # Auc3 row
        values = []
        for ds in pose_datasets:
            val = get_pose_metric(ds, "Auc3")
            if val is not None:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'Auc3':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in pose_datasets:
            val = get_pose_metric(ds, "Auc3")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        # Auc30 row
        values = []
        for ds in pose_datasets:
            val = get_pose_metric(ds, "Auc30")
            if val is not None:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'Auc30':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in pose_datasets:
            val = get_pose_metric(ds, "Auc30")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        # ============ DEPTH METRICS ============
        print(f"\n{Colors.BOLD_MAGENTA}📏 DEPTH ESTIMATION{Colors.RESET}")
        depth_datasets = ["hiroom", "eth3d", "7scenes", "scannetpp"]

        # Depth evaluation overview (MoGe-2 / MapAnything conventions):
        #   - Scale-invariant (median-aligned) depth: evaluates shape up-to-scale.
        #   - Affine-invariant depth: evaluates shape up-to-(scale+shift) (depth or disparity).
        #   - Metric depth: evaluates absolute depth in meters (no alignment).
        #   - Metric scale / coverage: reports global scale error and eval mask coverage.
        depth_rows = [
            # ---- scale-invariant (median-aligned) depth ----
            ("Scale-inv AbsRel (med)", "rel_depth", "abs_rel_scale_med"),
            ("Scale-inv δ@1.03 (med)", "rel_depth", "delta_1.03_scale_med"),
            ("Scale-inv δ@1.25 (med)", "rel_depth", "delta_1.25_scale_med"),

            # ---- affine-invariant depth (scale+shift) ----
            ("Affine-inv AbsRel (depth)", "rel_depth", "abs_rel_affine_depth"),
            ("Affine-inv δ@1.03 (depth)", "rel_depth", "delta_1.03_affine_depth"),
            ("Affine-inv δ@1.25 (depth)", "rel_depth", "delta_1.25_affine_depth"),
            ("Affine-inv AbsRel (disp)", "rel_depth", "abs_rel_affine_disp"),
            ("Affine-inv δ@1.03 (disp)", "rel_depth", "delta_1.03_affine_disp"),
            ("Affine-inv δ@1.25 (disp)", "rel_depth", "delta_1.25_affine_disp"),

            # ---- metric depth (no alignment) ----
            ("Metric AbsRel", "metric_depth", "abs_rel"),
            ("Metric δ@1.03", "metric_depth", "delta_1.03"),
            ("Metric δ@1.25", "metric_depth", "delta_1.25"),

            # ---- metric scale + coverage ----
            ("Median scale (s)", "metric_depth", "scale_med"),
            ("Metric scale rel", "metric_depth", "metric_scale_rel"),
            ("Valid pixels (%)", "metric_depth", "valid_pixels_pct"),
        ]

        metric_width = max(24, max(len(r[0]) for r in depth_rows) + 2)
        header = f"{'Metric':<{metric_width}}{'Avg':<{col_width}}"
        for ds in depth_datasets:
            if ds in DATASET_DISPLAY:
                header += f"{DATASET_DISPLAY[ds]:<{col_width}}"
        print("-" * len(strip_ansi(header)))
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print("-" * len(strip_ansi(header)))

        # Average computed over available datasets (ignoring N/A).
        for ridx, (label, mode, metric_name) in enumerate(depth_rows):
            values = [get_metric(ds, mode, metric_name) for ds in depth_datasets]
            valid_vals = [v for v in values if v is not None]
            avg = float(np.mean(valid_vals)) if valid_vals else None

            row = f"{label:<{metric_width}}{fmt_val(avg):<{col_width}}"
            for v in values:
                row += f"{fmt_val(v):<{col_width}}"
            print(row)
            # Visually separate depth metric groups.
            if ridx in (2, 8, 11):
                print("-" * len(strip_ansi(header)))

        # ============ RECON_UNPOSED METRICS ============
        print(f"\n{Colors.BOLD_MAGENTA}🏗️  RECON_UNPOSED (Pred Pose){Colors.RESET}")
        
        # For recon, exclude dtu64 from columns
        recon_datasets = ["hiroom", "eth3d", "dtu", "7scenes", "scannetpp"]
        avg_datasets = ["hiroom", "eth3d", "7scenes", "scannetpp"]  # Exclude DTU from avg
        
        # Header: Avg first, then datasets
        header = f"{'Metric':<15}{'Avg*':<{col_width}}"
        for ds in recon_datasets:
            header += f"{DATASET_DISPLAY[ds]:<{col_width}}"
        print("-" * len(strip_ansi(header)))
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print("-" * len(strip_ansi(header)))

        # F-score row (only metric for avg)
        values = []
        for ds in recon_datasets:
            val = get_metric(ds, "recon_unposed", "fscore")
            if val is not None and ds in avg_datasets:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'F-score':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in recon_datasets:
            val = get_metric(ds, "recon_unposed", "fscore")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        # Overall row (avg over 4 datasets excluding DTU)
        values = []
        for ds in recon_datasets:
            val = get_metric(ds, "recon_unposed", "overall")
            if val is not None and ds in avg_datasets:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'Overall':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in recon_datasets:
            val = get_metric(ds, "recon_unposed", "overall")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        # ============ RECON_POSED METRICS ============
        print(f"\n{Colors.BOLD_MAGENTA}🏗️  RECON_POSED (GT Pose){Colors.RESET}")
        
        # Header: Avg first, then datasets
        header = f"{'Metric':<15}{'Avg*':<{col_width}}"
        for ds in recon_datasets:
            header += f"{DATASET_DISPLAY[ds]:<{col_width}}"
        print("-" * len(strip_ansi(header)))
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print("-" * len(strip_ansi(header)))

        # F-score row (only metric for avg)
        values = []
        for ds in recon_datasets:
            val = get_metric(ds, "recon_posed", "fscore")
            if val is not None and ds in avg_datasets:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'F-score':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in recon_datasets:
            val = get_metric(ds, "recon_posed", "fscore")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        # Overall row (avg over 4 datasets excluding DTU)
        values = []
        for ds in recon_datasets:
            val = get_metric(ds, "recon_posed", "overall")
            if val is not None and ds in avg_datasets:
                values.append(val)
        avg = sum(values) / len(values) if values else None
        row = f"{'Overall':<15}{Colors.BOLD_GREEN}{fmt_val(avg):<{col_width}}{Colors.RESET}"
        for ds in recon_datasets:
            val = get_metric(ds, "recon_posed", "overall")
            row += f"{fmt_val(val):<{col_width}}"
        print(row)

        print(f"\n{Colors.CYAN}* Avg F-score / Overall = average over HiRoom, ETH3D, 7Scenes, ScanNet++ (4 datasets){Colors.RESET}")


def load_metrics_from_dir(metric_dir: str) -> TDict[str, dict]:
    """
    Load all metrics JSON files from a directory.

    Args:
        metric_dir: Path to directory containing metric JSON files

    Returns:
        Dictionary mapping filename (without .json) to metric data
    """
    metrics = {}
    if not os.path.exists(metric_dir):
        return metrics

    for filename in os.listdir(metric_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(metric_dir, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                # Handle trailing commas in JSON
                content = re.sub(r",\s*([\]\}])", r"\1", content)
                data = json.loads(content)
                key = filename[:-5]
                metrics[key] = data
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    return metrics


def main():
    """Command-line interface for metrics printing."""
    parser = argparse.ArgumentParser(
        description="Print DepthAnything3 benchmark evaluation metrics."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./eval_workspace/metric_results",
        help="Directory containing metric JSON files (comma-separated for comparison)",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Specific metric key to highlight",
    )
    args = parser.parse_args()

    # Support multiple directories for comparison
    input_dirs = [d.strip() for d in args.input_dir.split(",") if d.strip()]

    printer = MetricsPrinter(use_color=not args.no_color)

    if len(input_dirs) == 1:
        # Single directory - simple print
        metrics = load_metrics_from_dir(input_dirs[0])
        printer.print_results(metrics)
    else:
        # Multiple directories - comparison mode
        metrics_list = []
        labels = []
        for d in input_dirs:
            metrics = load_metrics_from_dir(d)
            if metrics:
                metrics_list.append(metrics)
                labels.append(os.path.basename(d.rstrip("/")))

        if metrics_list:
            printer.print_comparison(metrics_list, labels)
        else:
            print("No metrics found in specified directories")


if __name__ == "__main__":
    main()
