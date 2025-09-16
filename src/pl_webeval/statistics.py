"""
Statistical analysis and visualization module for PL_WebEval.
Generates comprehensive statistics and academic-quality visualizations after test runs.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class StatisticsGenerator:
    """Generate comprehensive statistical analysis and visualizations for PersonaLayer evaluations."""
    
    def __init__(self, run_dir: Path):
        """
        Initialize statistics generator.
        
        Args:
            run_dir: Directory containing test results
        """
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir / "statistical_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.agg_csv = self.results_dir / "aggregated_results.csv"
        self.persona_csv = self.results_dir / "persona_summary.csv"
        self.summary_md = self.results_dir / "results_summary.md"
        self.models_compare_csv = self.results_dir / "models_compare_pairs.csv"
        self.models_summary_md = self.results_dir / "models_compare_summary.md"
        self.delta_chart_png = self.results_dir / "delta_distribution.png"
        self.wcag_chart_png = self.results_dir / "wcag_compliance.png"
        
    def generate_all_statistics(self):
        """Generate all statistical outputs and visualizations."""
        try:
            logger.info(f"Generating statistics for run: {self.run_dir}")
            
            # Step 1: Aggregate results
            rows = self._aggregate_results()
            if not rows:
                logger.warning("No test results found to analyze")
                return
            
            # Step 2: Generate persona summary
            persona_summary = self._generate_persona_summary(rows)
            
            # Step 3: Compare models if multiple present
            models_used = self._detect_models(rows)
            if len(models_used) > 1:
                self._compare_models(rows)
            
            # Step 4: Generate visualizations
            self._create_visualizations(persona_summary, rows)
            
            # Step 5: Generate comprehensive markdown report
            self._generate_comprehensive_report(rows, persona_summary, models_used)
            
            logger.info(f"Statistical analysis complete. Results in: {self.results_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return False
    
    def _aggregate_results(self) -> List[Dict[str, Any]]:
        """Aggregate test results from individual test directories."""
        rows = []
        
        # Find all test directories
        for test_dir in self.run_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            test_info = test_dir / "test_info.txt"
            comp_json = test_dir / "comprehensive_analysis.json"
            
            if not test_info.exists():
                continue
            
            # Parse test info
            base_data = {
                "test_dir": str(test_dir),
                "site_id": test_dir.name,
            }
            
            # Parse test_info.txt
            info_data = self._parse_test_info(test_info)
            base_data.update(info_data)
            
            # Parse comprehensive_analysis.json if available
            if comp_json.exists():
                json_data = self._parse_json_metrics(comp_json)
                base_data.update(json_data)
            
            rows.append(base_data)
        
        # Write aggregated CSV
        if rows:
            fieldnames = sorted(set(k for row in rows for k in row.keys()))
            with open(self.agg_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Aggregated {len(rows)} test results to {self.agg_csv}")
        
        return rows
    
    def _parse_test_info(self, path: Path) -> Dict[str, Any]:
        """Parse test_info.txt file."""
        data = {
            "ux_profile": "",
            "bot_detection": None,
            "baseline_acc": None,
            "adapted_acc": None,
            "delta_acc": None,
            "adaptation_score": None,
            "adaptation_score_max": None,
            "wcag_adapted": None,
            "timestamp": "",
            "test_num": None,
            "url": "",
            "model": "",
        }
        
        patterns = {
            "ux_profile": re.compile(r"^UX Profile:\s*(.+)\s*$", re.I),
            "bot_detection": re.compile(r"^Bot Detection Encountered:\s*(True|False)\s*$", re.I),
            "baseline_acc": re.compile(r"^Baseline Accessibility Score:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.I),
            "adapted_acc": re.compile(r"^Adapted Accessibility Score:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.I),
            "adaptation_score": re.compile(r"^Adaptation Score:\s*([0-9]+)\s*/\s*([0-9]+)\s*$", re.I),
            "wcag_adapted": re.compile(r"^WCAG Compliance \(Adapted\):\s*([0-9]+(?:\.[0-9]+)?)%$", re.I),
            "timestamp": re.compile(r"^Timestamp:\s*(.+)\s*$", re.I),
            "test_num": re.compile(r"^Homepage Test\s+([0-9]+)\s*$", re.I),
            "url": re.compile(r"^URL:\s*(.+)\s*$", re.I),
            "model": re.compile(r"^Model:\s*(.+)\s*$", re.I),
        }
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    for key, pattern in patterns.items():
                        match = pattern.match(line)
                        if match:
                            if key == "ux_profile":
                                data["ux_profile"] = match.group(1).strip()
                            elif key == "bot_detection":
                                data["bot_detection"] = match.group(1).lower() == "true"
                            elif key == "baseline_acc":
                                data["baseline_acc"] = float(match.group(1))
                            elif key == "adapted_acc":
                                data["adapted_acc"] = float(match.group(1))
                            elif key == "adaptation_score":
                                data["adaptation_score"] = int(match.group(1))
                                data["adaptation_score_max"] = int(match.group(2))
                            elif key == "wcag_adapted":
                                data["wcag_adapted"] = float(match.group(1))
                            elif key == "timestamp":
                                data["timestamp"] = match.group(1).strip()
                            elif key == "test_num":
                                data["test_num"] = int(match.group(1))
                            elif key == "url":
                                data["url"] = match.group(1).strip()
                            elif key == "model":
                                data["model"] = match.group(1).strip()
                            break
            
            # Calculate delta
            if data["baseline_acc"] is not None and data["adapted_acc"] is not None:
                data["delta_acc"] = data["adapted_acc"] - data["baseline_acc"]
                
        except Exception as e:
            logger.error(f"Error parsing test_info.txt: {e}")
        
        return data
    
    def _parse_json_metrics(self, path: Path) -> Dict[str, Any]:
        """Parse comprehensive_analysis.json for additional metrics."""
        data = {}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            # Extract baseline/adapted metrics
            baseline = json_data.get("baseline_metrics", {})
            adapted = json_data.get("adapted_metrics", {})
            
            # Extract key metrics
            for prefix, metrics in [("baseline", baseline), ("adapted", adapted)]:
                for key in ["elements_count", "interactive_elements_count", 
                           "visual_complexity_score", "color_contrast_issues",
                           "text_readability_score", "adaptation_effectiveness_score"]:
                    if key in metrics and metrics[key] is not None:
                        data[f"{prefix}_{key}"] = metrics[key]
            
            # Calculate deltas
            for key in ["elements_count", "interactive_elements_count",
                       "visual_complexity_score", "color_contrast_issues",
                       "text_readability_score", "adaptation_effectiveness_score"]:
                baseline_key = f"baseline_{key}"
                adapted_key = f"adapted_{key}"
                if baseline_key in data and adapted_key in data:
                    try:
                        data[f"delta_{key}"] = float(data[adapted_key]) - float(data[baseline_key])
                    except:
                        pass
            
            # Extract models used
            models_used = json_data.get("models_used", {})
            if isinstance(models_used, dict):
                if models_used.get("generation"):
                    data["gen_model"] = models_used["generation"]
                if models_used.get("evaluation_metrics"):
                    data["eval_model_metrics"] = models_used["evaluation_metrics"]
                if models_used.get("evaluation_visual"):
                    data["eval_model_visual"] = models_used["evaluation_visual"]
                    
        except Exception as e:
            logger.error(f"Error parsing comprehensive_analysis.json: {e}")
        
        return data
    
    def _generate_persona_summary(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate per-persona summary statistics."""
        persona_groups = defaultdict(list)
        
        for row in rows:
            persona = row.get("ux_profile", "N/A")
            persona_groups[persona].append(row)
        
        persona_summary = []
        for persona, items in persona_groups.items():
            def safe_mean(vals):
                nums = [float(v) for v in vals if v is not None]
                return (sum(nums) / len(nums)) if nums else None
            
            summary = {
                "ux_profile": persona,
                "n_tests": len(items),
                "mean_baseline_accessibility": safe_mean([r.get("baseline_acc") for r in items]),
                "mean_adapted_accessibility": safe_mean([r.get("adapted_acc") for r in items]),
                "mean_delta_accessibility": safe_mean([r.get("delta_acc") for r in items]),
                "mean_adaptation_score": safe_mean([r.get("adaptation_score") for r in items]),
                "mean_wcag_adapted": safe_mean([r.get("wcag_adapted") for r in items]),
            }
            persona_summary.append(summary)
        
        # Sort by delta accessibility
        persona_summary.sort(key=lambda x: x.get("mean_delta_accessibility") or 0)
        
        # Write persona summary CSV
        fieldnames = ["ux_profile", "n_tests", "mean_baseline_accessibility",
                     "mean_adapted_accessibility", "mean_delta_accessibility",
                     "mean_adaptation_score", "mean_wcag_adapted"]
        
        with open(self.persona_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(persona_summary)
        
        logger.info(f"Generated persona summary for {len(persona_summary)} personas")
        return persona_summary
    
    def _detect_models(self, rows: List[Dict[str, Any]]) -> set:
        """Detect all models used in the evaluation."""
        models = set()
        for row in rows:
            for key in ["model", "gen_model", "eval_model_metrics", "eval_model_visual"]:
                model = row.get(key)
                if model:
                    models.add(model)
        return models
    
    def _compare_models(self, rows: List[Dict[str, Any]]):
        """Compare performance across different models."""
        # Group by model
        model_groups = defaultdict(list)
        for row in rows:
            model = row.get("eval_model_visual") or row.get("model", "unknown")
            model_groups[model].append(row)
        
        # Calculate per-model statistics
        model_stats = []
        for model, items in model_groups.items():
            def safe_mean(vals):
                nums = [float(v) for v in vals if v is not None]
                return (sum(nums) / len(nums)) if nums else None
            
            stats = {
                "model": model,
                "n_tests": len(items),
                "mean_baseline": safe_mean([r.get("baseline_acc") for r in items]),
                "mean_adapted": safe_mean([r.get("adapted_acc") for r in items]),
                "mean_delta": safe_mean([r.get("delta_acc") for r in items]),
                "mean_wcag": safe_mean([r.get("wcag_adapted") for r in items]),
                "mean_adaptation_score": safe_mean([r.get("adaptation_score") for r in items]),
            }
            model_stats.append(stats)
        
        # Write model comparison summary
        with open(self.models_summary_md, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Summary\n\n")
            f.write(f"Total models compared: {len(model_stats)}\n\n")
            f.write("## Per-Model Performance\n\n")
            f.write("| Model | Tests | Baseline | Adapted | Delta | WCAG | Adaptation Score |\n")
            f.write("|-------|-------|----------|---------|-------|------|------------------|\n")
            
            for stats in model_stats:
                f.write(f"| {stats['model']} ")
                f.write(f"| {stats['n_tests']} ")
                f.write(f"| {stats['mean_baseline']:.1f}% " if stats['mean_baseline'] else "| N/A ")
                f.write(f"| {stats['mean_adapted']:.1f}% " if stats['mean_adapted'] else "| N/A ")
                f.write(f"| {stats['mean_delta']:+.1f}pp " if stats['mean_delta'] else "| N/A ")
                f.write(f"| {stats['mean_wcag']:.1f}% " if stats['mean_wcag'] else "| N/A ")
                f.write(f"| {stats['mean_adaptation_score']:.1f} " if stats['mean_adaptation_score'] else "| N/A ")
                f.write("|\n")
        
        logger.info(f"Generated model comparison for {len(model_stats)} models")
    
    def _create_visualizations(self, persona_summary: List[Dict[str, Any]], rows: List[Dict[str, Any]]):
        """Create academic-quality visualizations."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not installed. Skipping visualizations. Install with: pip install matplotlib")
            return
        
        # Create delta distribution chart
        self._create_delta_chart(persona_summary, plt)
        
        # Create WCAG compliance chart
        self._create_wcag_chart(persona_summary, plt)
        
        logger.info("Generated visualization charts")
    
    def _create_delta_chart(self, persona_summary: List[Dict[str, Any]], plt):
        """Create bar chart showing accessibility delta per persona."""
        names = []
        deltas = []
        
        for row in persona_summary:
            name = row.get("ux_profile", "").strip()
            delta = row.get("mean_delta_accessibility")
            if name and delta is not None:
                names.append(name)
                deltas.append(delta)
        
        if not names:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors: negative = red, positive = green
        colors = ["#d32f2f" if d < 0 else "#388e3c" for d in deltas]
        
        # Create bars
        x = range(len(names))
        bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, deltas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8)
        
        # Formatting
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Accessibility Improvement (percentage points)", fontsize=11)
        ax.set_title("PersonaLayer: Accessibility Improvements by UX Profile", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.delta_chart_png, dpi=200, bbox_inches="tight")
        plt.close()
    
    def _create_wcag_chart(self, persona_summary: List[Dict[str, Any]], plt):
        """Create bar chart showing WCAG compliance per persona."""
        names = []
        baseline_vals = []
        adapted_vals = []
        
        for row in persona_summary:
            name = row.get("ux_profile", "").strip()
            baseline = row.get("mean_baseline_accessibility")
            adapted = row.get("mean_adapted_accessibility")
            
            if name and baseline is not None and adapted is not None:
                names.append(name)
                baseline_vals.append(baseline)
                adapted_vals.append(adapted)
        
        if not names:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar positions
        x = range(len(names))
        width = 0.35
        
        # Create grouped bars
        bars1 = ax.bar([i - width/2 for i in x], baseline_vals, width, 
                      label='Baseline', color='#ff7043', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar([i + width/2 for i in x], adapted_vals, width,
                      label='Adapted', color='#66bb6a', edgecolor='black', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel("UX Profile", fontsize=11)
        ax.set_ylabel("Accessibility Score (%)", fontsize=11)
        ax.set_title("PersonaLayer: Baseline vs Adapted Accessibility Scores", fontsize=12, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement indicators
        for i, (b, a) in enumerate(zip(baseline_vals, adapted_vals)):
            if a > b:
                ax.annotate('', xy=(i, a), xytext=(i, b),
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        
        plt.tight_layout()
        plt.savefig(self.wcag_chart_png, dpi=200, bbox_inches="tight")
        plt.close()
    
    def _generate_comprehensive_report(self, rows: List[Dict[str, Any]], 
                                      persona_summary: List[Dict[str, Any]],
                                      models_used: set):
        """Generate comprehensive markdown report."""
        with open(self.summary_md, "w", encoding="utf-8") as f:
            f.write("# PersonaLayer Evaluation Results Summary\n\n")
            f.write("## Overview\n\n")
            f.write(f"- **Total tests completed:** {len(rows)}\n")
            f.write(f"- **UX profiles evaluated:** {len(persona_summary)}\n")
            f.write(f"- **Models used:** {', '.join(sorted(models_used)) if models_used else 'N/A'}\n")
            f.write(f"- **Test directory:** `{self.run_dir}`\n\n")
            
            # Overall statistics
            all_deltas = [r.get("delta_acc") for r in rows if r.get("delta_acc") is not None]
            if all_deltas:
                f.write("## Overall Performance\n\n")
                f.write(f"- **Mean accessibility improvement:** {sum(all_deltas)/len(all_deltas):.2f} percentage points\n")
                f.write(f"- **Maximum improvement:** {max(all_deltas):.2f} pp\n")
                f.write(f"- **Minimum improvement:** {min(all_deltas):.2f} pp\n")
                f.write(f"- **Tests with positive improvement:** {sum(1 for d in all_deltas if d > 0)}/{len(all_deltas)}\n\n")
            
            # Per-persona results
            f.write("## Persona-wise Results\n\n")
            f.write("| UX Profile | Tests | Baseline | Adapted | Improvement | WCAG | Adaptation Score |\n")
            f.write("|------------|-------|----------|---------|-------------|------|------------------|\n")
            
            for row in persona_summary:
                f.write(f"| {row['ux_profile']} ")
                f.write(f"| {row['n_tests']} ")
                
                baseline = row.get('mean_baseline_accessibility')
                f.write(f"| {baseline:.1f}% " if baseline is not None else "| N/A ")
                
                adapted = row.get('mean_adapted_accessibility')
                f.write(f"| {adapted:.1f}% " if adapted is not None else "| N/A ")
                
                delta = row.get('mean_delta_accessibility')
                if delta is not None:
                    f.write(f"| {delta:+.1f} pp ")
                else:
                    f.write("| N/A ")
                
                wcag = row.get('mean_wcag_adapted')
                f.write(f"| {wcag:.1f}% " if wcag is not None else "| N/A ")
                
                score = row.get('mean_adaptation_score')
                f.write(f"| {score:.1f} " if score is not None else "| N/A ")
                
                f.write("|\n")
            
            # Top performing personas
            f.write("\n## Key Findings\n\n")
            
            # Sort by improvement
            sorted_personas = sorted(persona_summary, 
                                   key=lambda x: x.get('mean_delta_accessibility') or 0, 
                                   reverse=True)
            
            if len(sorted_personas) >= 3:
                f.write("### Top 3 Personas by Improvement\n\n")
                for i, p in enumerate(sorted_personas[:3], 1):
                    delta = p.get('mean_delta_accessibility')
                    if delta is not None:
                        f.write(f"{i}. **{p['ux_profile']}**: +{delta:.1f} pp improvement\n")
                
                f.write("\n### Bottom 3 Personas by Improvement\n\n")
                for i, p in enumerate(sorted_personas[-3:], 1):
                    delta = p.get('mean_delta_accessibility')
                    if delta is not None:
                        f.write(f"{i}. **{p['ux_profile']}**: {delta:+.1f} pp change\n")
            
            # Visualization references
            f.write("\n## Visualizations\n\n")
            f.write("- [Accessibility Improvement Chart](delta_distribution.png)\n")
            f.write("- [WCAG Compliance Comparison](wcag_compliance.png)\n")
            
            if (self.models_summary_md).exists():
                f.write("- [Model Comparison Analysis](models_compare_summary.md)\n")
            
            # Data files
            f.write("\n## Data Files\n\n")
            f.write("- **Aggregated results:** `aggregated_results.csv`\n")
            f.write("- **Persona summary:** `persona_summary.csv`\n")
            if (self.models_compare_csv).exists():
                f.write("- **Model comparison:** `models_compare_pairs.csv`\n")
            
            f.write("\n---\n")
            f.write("*Generated by PL_WebEval Statistical Analysis Module*\n")
        
        logger.info(f"Generated comprehensive report: {self.summary_md}")


def generate_statistics(run_dir: str):
    """
    Main entry point for generating statistics.
    
    Args:
        run_dir: Directory containing test results
    """
    generator = StatisticsGenerator(Path(run_dir))
    return generator.generate_all_statistics()