from datetime import datetime
from typing import List, Optional
from pathlib import Path
import base64
import csv
# Make sure 'data_models.py' is in the same directory or accessible via PYTHONPATH
from .data_models import HomepageResult

def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """Encode an image file to base64 string."""
    if not image_path.exists():
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        return None

def load_persona_summary(csv_path: Path) -> List[dict]:
    """Load persona summary data from CSV."""
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def generate_comprehensive_report_html(results: List[HomepageResult], stats_dir: Optional[Path] = None) -> str:
    """Generates a comprehensive HTML report from a list of test results with optional statistical analysis."""

    if not results:
        return "<h1>No test results available</h1>"

    total_tests = len(results)
    successful_tests = len([r for r in results if r.status == "success"])
    blocked_tests = len([r for r in results if r.status == "blocked"])
    failed_tests = len([r for r in results if r.status == "error"])

    total_tokens = sum(sum(m.total_tokens for m in result.llm_metrics) for result in results if result.llm_metrics)
    adapted_results_count = len([r for r in results if r.ux_adaptations])

    # Load statistical data if available
    stats_section = ""
    if stats_dir and stats_dir.exists():
        persona_summary = load_persona_summary(stats_dir / "persona_summary.csv")
        delta_chart = encode_image_to_base64(stats_dir / "delta_distribution.png")
        wcag_chart = encode_image_to_base64(stats_dir / "wcag_compliance.png")
        models_md = None
        try:
            models_md_path = stats_dir / "models_compare_summary.md"
            if models_md_path.exists():
                models_md = models_md_path.read_text(encoding="utf-8")
        except Exception:
            models_md = None
        
        if persona_summary or delta_chart or wcag_chart or models_md:
            stats_section = generate_statistics_section(persona_summary, delta_chart, wcag_chart, models_md)

    # --- HTML Header and Styles ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PersonaLayer Web Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 3px solid #667eea; padding-bottom: 30px; }}
        h1 {{ color: #2d3748; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #718096; font-size: 1.2em; margin-bottom: 15px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: white; margin-bottom: 8px; }}
        .metric-label {{ color: #e2e8f0; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }}
        
        /* Statistical Analysis Section */
        .stats-section {{ background: #f7fafc; padding: 30px; border-radius: 12px; margin: 40px 0; }}
        .stats-header {{ text-align: center; color: #2d3748; margin-bottom: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }}
        .stats-chart {{ text-align: center; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats-chart img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .stats-table {{ width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats-table th {{ background: #667eea; color: white; padding: 12px; text-align: left; font-weight: 600; }}
        .stats-table td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
        .stats-table tr:hover {{ background: #f7fafc; }}
        .positive {{ color: #48bb78; font-weight: bold; }}
        .negative {{ color: #f56565; font-weight: bold; }}
        
        .test-result {{ border: 1px solid #e2e8f0; margin-bottom: 30px; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .test-header {{ background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%); padding: 20px; border-bottom: 2px solid #e2e8f0; }}
        .test-header h3 {{ margin: 0 0 10px 0; color: #2d3748; font-size: 1.4em; }}
        .test-header p {{ margin: 0; font-size: 0.95em; color: #4a5568; }}
        .test-content {{ padding: 25px; }}
        .status-success {{ border-left: 6px solid #48bb78; }}
        .status-blocked {{ border-left: 6px solid #ed8936; }}
        .status-error {{ border-left: 6px solid #f56565; }}
        .screenshot {{ max-width: 100%; height: auto; border: 2px solid #e2e8f0; border-radius: 8px; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .screenshot-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin: 30px 0; }}
        .screenshot-box {{ text-align: center; background: #f7fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .screenshot-label {{ font-weight: bold; margin-bottom: 15px; color: #2d3748; display: block; font-size: 1.1em; }}
        .analysis-section {{ margin-top: 25px; }}
        .expert-analysis, .visual-indicators, .bot-detection-details, .llm-reasoning, .expert-section, .visual-diff-item, .adaptations-list {{ 
            background: #f7fafc; padding: 20px; border-radius: 8px; margin: 15px 0; border: 1px solid #e2e8f0; 
        }}
        .adaptation-score {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }}
        .score-0 {{ background: #feb2b2; color: #742a2a; }}
        .score-1 {{ background: #fbd38d; color: #744210; }}
        .score-2 {{ background: #9ae6b4; color: #22543d; }}
        .detailed-notes pre, .llm-reasoning pre, .bot-detection-details pre, .adaptations-list pre {{ 
            background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 8px; white-space: pre-wrap; 
            word-wrap: break-word; font-family: 'Courier New', Courier, monospace; font-size: 0.9em; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .css-js-analysis {{ background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 8px; margin: 15px 0; font-family: monospace; font-size: 0.95em; overflow-x: auto; }}
        .expert-title {{ font-weight: bold; color: #2d3748; margin-bottom: 10px; display: block; font-size: 1.1em; }}
        .collapsible {{ 
            cursor: pointer; padding: 14px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; border: none; width: 100%; text-align: left; outline: none; font-size: 1.1em; 
            border-radius: 8px; margin: 20px 0 8px 0; transition: all 0.3s ease; font-weight: 600;
        }}
        .collapsible:hover {{ background: linear-gradient(135deg, #5a67d8 0%, #6b4199 100%); transform: translateX(5px); }}
        .collapsible::after {{ content: '\\002B'; color: white; font-weight: bold; float: right; margin-left: 5px; }}
        .collapsible.active::after {{ content: "\\2212"; }}
        .collapsible-content {{ 
            padding: 0 20px; display: none; overflow: hidden; background-color: #f7fafc; margin-bottom: 15px; 
            border-radius: 0 0 8px 8px; border: 1px solid #e2e8f0; border-top: none; 
        }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ padding: 12px; border: 1px solid #e2e8f0; text-align: left; }}
        .metrics-table th {{ background-color: #edf2f7; font-weight: bold; color: #2d3748; }}
        .metrics-table tr:hover {{ background: #f7fafc; }}
        .error-message {{ background: #feb2b2; color: #742a2a; padding: 20px; border-radius: 8px; border: 2px solid #fc8181; }}
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            var coll = document.getElementsByClassName("collapsible");
            for (var i = 0; i < coll.length; i++) {{
                coll[i].addEventListener("click", function() {{
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.display === "block") {{
                        content.style.display = "none";
                    }} else {{
                        content.style.display = "block";
                    }}
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 PersonaLayer Web Evaluation Report</h1>
            <div class="subtitle">Profile-Informed Web Personalization for Enhanced User Experience</div>
            <p style="color: #718096;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{successful_tests}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{blocked_tests}</div>
                <div class="metric-label">Bot Blocked</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{failed_tests}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{adapted_results_count}</div>
                <div class="metric-label">Adapted</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_tokens:,}</div>
                <div class="metric-label">Tokens</div>
            </div>
        </div>

        {stats_section}
"""

    # --- Loop Through Each Test Result ---
    for i, result in enumerate(results, 1):
        status_class = f"status-{result.status}"

        html_content += f"""
        <div class="test-result {status_class}">
            <div class="test-header">
                <h3>Test {i}: {result.test_case.website}</h3>
                <p>
                    <strong>Status:</strong> {result.status.upper()} |
                    <strong>UX Profile:</strong> {result.test_case.ux_profile or 'None'} |
                    <strong>Execution Time:</strong> {result.execution_time:.2f}s |
                    <strong>Adaptations:</strong> {len(result.ux_adaptations)}
                </p>
            </div>
            <div class="test-content">
"""

        # --- Screenshots ---
        if result.baseline_screenshot or result.adapted_screenshot:
            html_content += '<div class="screenshot-container">'
            if result.baseline_screenshot:
                html_content += f"""
                <div class="screenshot-box">
                    <span class="screenshot-label">📸 Baseline Screenshot</span>
                    <img src="data:image/png;base64,{result.baseline_screenshot}" class="screenshot" alt="Baseline Screenshot">
                </div>
"""
            if result.adapted_screenshot and result.test_case.ux_profile:
                html_content += f"""
                <div class="screenshot-box">
                    <span class="screenshot-label">✨ Adapted Screenshot ({result.test_case.ux_profile})</span>
                    <img src="data:image/png;base64,{result.adapted_screenshot}" class="screenshot" alt="Adapted Screenshot">
                </div>
"""
            html_content += '</div>'

        # --- Bot Detection Details ---
        if result.bot_detection_encountered and result.bot_detection_result:
            bot = result.bot_detection_result
            confidence = bot.confidence_score * 100
            score_cls_bot = '2' if bot.confidence_score > 0.8 else '1' if bot.confidence_score > 0.5 else '0'
            html_content += f"""
                <div class="bot-detection-details">
                    <h4>🤖 Bot Detection Analysis</h4>
                    <p><strong>Type:</strong> {bot.detection_type} | <strong>Confidence:</strong> <span class="adaptation-score score-{score_cls_bot}">{confidence:.1f}%</span></p>
                    <p><strong>Action:</strong> {bot.recommended_action}</p>
                    <button type="button" class="collapsible" onclick="toggleCollapsible(this)">LLM Reasoning</button>
                    <div class="collapsible-content">
                        <pre>{bot.llm_reasoning}</pre>
                    </div>
                    <button type="button" class="collapsible" onclick="toggleCollapsible(this)">Technical Observations</button>
                    <div class="collapsible-content">
                        <pre>{bot.technical_observations}</pre>
                    </div>
                </div>
"""

        # --- UX Adaptations ---
        if result.ux_adaptations:
            html_content += """
                <button type="button" class="collapsible" onclick="toggleCollapsible(this)">🔧 UX Adaptations Applied</button>
                <div class="collapsible-content">
                    <div class="adaptations-list">
"""
            for adapt in result.ux_adaptations:
                html_content += f"""
                        <div>
                            <strong>{adapt.get('type', 'N/A').upper()}:</strong> {adapt.get('description', 'N/A')}
                            <pre>{adapt.get('code', 'No code')}</pre>
                        </div>
"""
            html_content += """
                    </div>
                </div>
"""

        # --- Scientific Metrics ---
        if result.baseline_metrics or result.adapted_metrics:
            html_content += """
                <button type="button" class="collapsible" onclick="toggleCollapsible(this)">📊 Scientific Metrics</button>
                <div class="collapsible-content">
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Baseline</th><th>Adapted</th><th>Change</th></tr>
"""
            metrics_map = [
                ("Accessibility Score", "accessibility_score", "%.1f"),
                ("Visual Complexity", "visual_complexity_score", "%.1f"),
                ("Text Readability", "text_readability_score", "%.1f"),
                ("Color Contrast Issues", "color_contrast_issues", "%d"),
                ("Elements Count", "elements_count", "%d"),
                ("Interactive Elements", "interactive_elements_count", "%d")
            ]
            for name, attr, fmt in metrics_map:
                base_val = getattr(result.baseline_metrics, attr, 0) if result.baseline_metrics else 0
                adap_val = getattr(result.adapted_metrics, attr, None) if result.adapted_metrics else None
                base_str = (fmt % base_val) if isinstance(base_val, (int, float)) else 'N/A'
                adap_str = (fmt % adap_val) if adap_val is not None else 'N/A'
                change_str = 'N/A'
                change_class = ''
                if adap_val is not None and isinstance(base_val, (int, float)):
                    change = adap_val - base_val
                    if attr in ["accessibility_score", "text_readability_score"]:
                        change_class = 'positive' if change > 0 else 'negative' if change < 0 else ''
                    elif attr in ["color_contrast_issues", "visual_complexity_score"]:
                        change_class = 'positive' if change < 0 else 'negative' if change > 0 else ''
                    change_str = (f'{change:+.1f}' if isinstance(change, float) else f'{change:+d}')
                html_content += f'<tr><td>{name}</td><td>{base_str}</td><td>{adap_str}</td><td class="{change_class}">{change_str}</td></tr>'
            html_content += """
                    </table>
                </div>
"""

        # --- Visual Comparison ---
        if result.visual_comparison:
            vc = result.visual_comparison
            ad_score = vc.adaptation_score
            score_cls_vc = f"score-{ad_score.score}" if ad_score else "score-0"
            html_content += f"""
                <div class="analysis-section">
                    <h4>🔬 Comprehensive Visual Analysis</h4>
                    <div class="expert-section">
                        <p><strong>Adaptation Score:</strong> <span class="adaptation-score {score_cls_vc}">{ad_score.score if ad_score else 'N/A'}/2 - {ad_score.reasoning if ad_score else 'N/A'}</span></p>
                        <p><strong>WCAG Compliance:</strong> {ad_score.wcag_compliance_score if ad_score else 'N/A'}% | <strong>Improvement Score:</strong> {vc.improvement_score}/10</p>
                    </div>
                    <button type="button" class="collapsible" onclick="toggleCollapsible(this)">Expert Analysis Summary</button>
                    <div class="collapsible-content">
"""
            if vc.expert_analysis:
                ea = vc.expert_analysis
                if ea.accessibility_expert:
                    html_content += f'<div class="expert-section"><span class="expert-title">♿ Accessibility</span><p>{ea.accessibility_expert.get("detailed_reasoning", "N/A")}</p></div>'
                if ea.ux_expert:
                    html_content += f'<div class="expert-section"><span class="expert-title">🎨 UX</span><p>{ea.ux_expert.get("user_journey_impact", "N/A")}</p></div>'
                if ea.visual_critic:
                    html_content += f'<div class="expert-section"><span class="expert-title">🎯 Visual</span><p>{ea.visual_critic.get("design_critique", "N/A")}</p></div>'
                if ea.css_analysis and not ea.css_analysis.get('error'):
                    html_content += f'<div class="expert-section"><span class="expert-title">💻 CSS</span><p>Responsive: {ea.css_analysis.get("responsive_design", "N/A")}</p></div>'
                if ea.js_analysis and not ea.js_analysis.get('error'):
                     html_content += f'<div class="expert-section"><span class="expert-title">🔌 JS</span><p>Performance: {ea.js_analysis.get("performance_impact", "N/A")}</p></div>'

            html_content += """
                    </div>
                    <button type="button" class="collapsible" onclick="toggleCollapsible(this)">Visual Differences</button>
                    <div class="collapsible-content">
"""
            for diff in vc.visual_differences[:5]:
                html_content += f'<div class="visual-diff-item"><strong>{diff.get("category", "N/A")}:</strong> {diff.get("description", "N/A")} <em>(Impact: {diff.get("impact", "N/A")})</em></div>'
            html_content += """
                    </div>
                    <button type="button" class="collapsible" onclick="toggleCollapsible(this)">Detailed Notes</button>
                    <div class="collapsible-content">
                        <pre>{vc.detailed_notes}</pre>
                    </div>
                </div>
"""

        # --- LLM Metrics ---
        if result.llm_metrics:
            tokens = sum(m.total_tokens for m in result.llm_metrics)
            html_content += f"""
                <button type="button" class="collapsible" onclick="toggleCollapsible(this)">📊 LLM Usage ({tokens:,} tokens)</button>
                <div class="collapsible-content">
                    <table class="metrics-table">
                        <tr><th>Type</th><th>Model</th><th>Tokens</th></tr>
"""
            for m in result.llm_metrics:
                html_content += f'<tr><td>{m.analysis_type}</td><td>{m.model}</td><td>{m.total_tokens:,}</td></tr>'
            html_content += """
                    </table>
                </div>
"""

        # --- Error Message ---
        if result.error_message:
            html_content += f"""
                <div class="error-message">
                    <strong>⚠️ Error:</strong> <pre>{result.error_message}</pre>
                </div>
"""

        html_content += """
            </div>
        </div>
"""

    # --- HTML Footer ---
    html_content += """
    </div>
</body>
</html>
"""
    return html_content


def generate_statistics_section(persona_summary: List[dict], delta_chart: Optional[str], wcag_chart: Optional[str], models_md: Optional[str] = None) -> str:
    """Generate the statistical analysis section of the HTML report, including charts, persona table, and optional model comparison."""
    
    html = """
        <div class="stats-section">
            <h2 class="stats-header">📊 Statistical Analysis</h2>
    """
    
    # Add charts if available
    if delta_chart or wcag_chart:
        html += '<div class="stats-grid">'
        
        if delta_chart:
            html += f"""
                <div class="stats-chart">
                    <h3>Accessibility Improvements by Persona</h3>
                    <img src="data:image/png;base64,{delta_chart}" alt="Delta Distribution Chart">
                </div>
            """
        
        if wcag_chart:
            html += f"""
                <div class="stats-chart">
                    <h3>WCAG Compliance Comparison</h3>
                    <img src="data:image/png;base64,{wcag_chart}" alt="WCAG Compliance Chart">
                </div>
            """
        
        html += '</div>'
    
    # Add persona summary table
    if persona_summary:
        html += """
            <h3 style="text-align: center; margin-top: 30px;">Persona Performance Summary</h3>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>UX Profile</th>
                        <th>Tests</th>
                        <th>Baseline</th>
                        <th>Adapted</th>
                        <th>Improvement</th>
                        <th>WCAG</th>
                        <th>Adaptation Score</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for row in persona_summary:
            improvement = row.get('mean_delta_accessibility', '')
            improvement_class = ''
            if improvement:
                try:
                    val = float(improvement)
                    improvement_class = 'positive' if val > 0 else 'negative' if val < 0 else ''
                    improvement = f"{val:+.1f} pp"
                except:
                    improvement = 'N/A'
            
            html += f"""
                <tr>
                    <td><strong>{row.get('ux_profile', 'N/A')}</strong></td>
                    <td>{row.get('n_tests', 'N/A')}</td>
                    <td>{float(row.get('mean_baseline_accessibility', 0)):.1f}%</td>
                    <td>{float(row.get('mean_adapted_accessibility', 0)):.1f}%</td>
                    <td class="{improvement_class}">{improvement}</td>
                    <td>{float(row.get('mean_wcag_adapted', 0)):.1f}%</td>
                    <td>{float(row.get('mean_adaptation_score', 0)):.1f}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        """
    
    # Add model comparison (markdown rendered plainly) if available
    if models_md:
        # Basic HTML escaping for safety; since this is a local report, keep it minimal
        try:
            import html as _html
            safe_md = _html.escape(models_md)
        except Exception:
            safe_md = models_md
        html += f"""
            <h3 style="text-align: center; margin-top: 30px;">Model Comparison Summary</h3>
            <div class="stats-table" style="padding: 0;">
                <div style="background: #1a202c; color: #e2e8f0; padding: 16px; font-family: monospace; white-space: pre-wrap; overflow-x: auto;">
{safe_md}
                </div>
            </div>
        """
    
    html += """
        </div>
    """
    
    return html


def generate_html_report(results: List[HomepageResult], output_path: Path):
    """Generate and save an HTML report with optional statistical analysis."""
    
    # Check if statistical analysis exists
    stats_dir = output_path.parent / "statistical_analysis"
    
    # Generate HTML content
    html_content = generate_comprehensive_report_html(results, stats_dir if stats_dir.exists() else None)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)