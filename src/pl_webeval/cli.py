"""
PersonaLayer WebEval CLI - Command-line interface for web evaluation tool
Part of the PersonaLayer research project for profile-informed web personalization
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List
import csv
from datetime import datetime
import hashlib

from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pl_webeval.evaluator import WebEvaluator
from pl_webeval.data_models import TestCase
from pl_webeval.state_manager import StateManager, CheckpointManager
from pl_webeval.statistics import generate_statistics

# Import comprehensive analysis functionality
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
try:
    from analyze_results import TestResultsAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Load environment variables
load_dotenv()


def load_test_cases(csv_path: Path) -> tuple:
    """Load test cases from CSV file and compute hash for change detection."""
    test_cases = []
    
    if not csv_path.exists():
        print(f"❌ Test cases CSV not found at {csv_path}")
        return [], ""
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    website = row.get('Website') or row.get('website')
                    if not website:
                        print(f"Skipping row {row_num}: 'Website' column is missing or empty.")
                        continue
                    
                    # Load ALL test cases - state manager will handle skipping
                    tc = TestCase(
                        website=website,
                        task_goal=row.get('Task Goal', f'Analyze {website}'),
                        task_steps=row.get('Task Steps', 'Load page and perform analysis.'),
                        success_criteria=row.get('Success Criteria', 'Page loads and analysis completes.'),
                        llm_model=row.get('LLM') or row.get('llm_model', 'openai/gpt-4o-mini'),
                        ux_profile=row.get('Ux Profile') or row.get('ux_profile', '')
                    )
                    
                    # Attach stable TestCaseID from CSV
                    setattr(tc, "case_id", row.get('TestCaseID', f"TC_{row_num:04d}"))
                    
                    # Attach generation/evaluation models if provided
                    gen_csv = (row.get('Gen LLM') or row.get('GenLLM') or "").strip()
                    eval_csv = (row.get('LLM') or row.get('llm_model') or "").strip()
                    setattr(tc, "gen_llm_model", gen_csv if gen_csv else tc.llm_model)
                    setattr(tc, "eval_llm_model", eval_csv if eval_csv else tc.llm_model)
                    
                    test_cases.append(tc)
                    
                except KeyError as e:
                    print(f"Skipping row {row_num} due to missing key: {e}")
                except Exception as e:
                    print(f"Error processing row {row_num}: {e}")
    
    except Exception as e:
        print(f"Error reading test cases CSV: {e}")
    
    # Compute hash for change detection
    test_hash = StateManager.compute_test_cases_hash(test_cases)
    
    return test_cases, test_hash


async def main():
    """Main execution function for PersonaLayer WebEval."""
    
    parser = argparse.ArgumentParser(
        description="PersonaLayer WebEval - Profile-Informed Web Personalization Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pl_webeval.cli --testcases data/test_cases_sample.csv --output results/run1
  python -m pl_webeval.cli --testcases data/test_cases.csv --output results/experiment1
        """
    )
    
    parser.add_argument(
        "--testcases",
        type=str,
        help="Path to test cases CSV file",
        default=None
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results",
        default=None
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
        default=None
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from previous state",
        default=False
    )
    
    parser.add_argument(
        "--analyze-only",
        type=str,
        help="Only run analysis on existing results directory (skip test execution)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Handle analyze-only mode
    if args.analyze_only:
        if ANALYSIS_AVAILABLE:
            analyze_dir = Path(args.analyze_only)
            if not analyze_dir.is_absolute():
                analyze_dir = Path.cwd() / analyze_dir
            
            if not analyze_dir.exists():
                print(f"❌ Results directory not found: {analyze_dir}")
                return 1
            
            print("=" * 60)
            print("🔬 PersonaLayer WebEval - Analysis Mode")
            print("=" * 60)
            print(f"📁 Analyzing results in: {analyze_dir}")
            print("=" * 60)
            
            try:
                analyzer = TestResultsAnalyzer(analyze_dir)
                analyzer.run_full_analysis()
                print("\n✅ Analysis complete!")
                return 0
            except Exception as e:
                print(f"\n❌ Analysis failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print("❌ Analysis module not available. Ensure analyze_results.py is in scripts/")
            return 1
    
    # Get API key (only needed for test execution)
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found. Please set it as environment variable or pass with --api-key")
        return 1
    
    # Determine paths
    pkg_root = Path(__file__).resolve().parents[2]  # PL_WebEval directory
    
    # Test cases CSV
    if args.testcases:
        csv_path = Path(args.testcases)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
    else:
        # Use default test cases file
        csv_path = pkg_root / "data" / "test_cases_default.csv"
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = pkg_root / "results" / f"run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔬 PersonaLayer WebEval - Web Personalization Evaluation")
    print("=" * 60)
    print(f"📂 Test cases: {csv_path}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Load test cases with hash
    test_cases, test_hash = load_test_cases(csv_path)
    if not test_cases:
        print("❌ No test cases loaded. Exiting.")
        return 1
    
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Initialize state manager for robust state tracking
    state_manager = StateManager(output_dir, test_hash)
    state_manager.set_total_tests(len(test_cases))
    
    # Check for resume (unless --no-resume flag is set)
    if not args.no_resume:
        resume_point = state_manager.get_resume_point()
        if resume_point:
            print(f"📌 Resuming from previous run...")
            progress = state_manager.get_progress_summary()
            print(f"   Completed: {progress['completed']}")
            print(f"   Failed: {progress['failed']}")
            print(f"   Remaining: {progress['remaining']}")
    else:
        # Clear state for fresh start
        state_manager.state = state_manager._create_new_state()
        state_manager.state['test_cases_hash'] = test_hash
        state_manager.set_total_tests(len(test_cases))
        print("🔄 Starting fresh evaluation (--no-resume flag set)")
    
    # Create evaluator
    evaluator = WebEvaluator(api_key, output_dir)
    evaluator.state_manager = state_manager  # Pass state manager to evaluator
    
    # Run tests with state management
    try:
        print("\n🚀 Starting evaluation suite...")
        results = []
        
        # Connect to Playwright MCP once at the start
        # This matches ModularV2's approach and avoids async context issues
        try:
            await evaluator.connect_to_playwright_mcp()
            if not evaluator.session:
                print("⚠️ Could not establish Playwright MCP connection, but continuing...")
        except Exception as e:
            print(f"⚠️ Playwright MCP connection warning: {e}")
        
        for i, test_case in enumerate(test_cases, 1):
            # Create unique test ID
            test_id = f"{test_case.website}_{test_case.ux_profile}_{test_case.llm_model}"
            
            # Check if already completed
            if state_manager.should_skip(test_id):
                print(f"⏭️  Skipping test {i}/{len(test_cases)} (already completed)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(test_cases)}")
            print(f"Website: {test_case.website}")
            print(f"UX Profile: {test_case.ux_profile}")
            print(f"LLM Model: {test_case.llm_model}")
            print(f"{'='*60}")
            
            # Mark as in progress
            state_manager.mark_in_progress(test_id)
            
            try:
                # Run single test using execute_test method
                result = await evaluator.execute_test(test_case)
                
                # Mark as completed
                state_manager.mark_completed(test_id, {
                    'status': result.status,
                    'wcag_score': result.wcag_score if hasattr(result, 'wcag_score') else None,
                    'adaptations_count': len(result.ux_adaptations) if result.ux_adaptations else 0
                })
                
                results.append(result)
                print(f"✓ Test completed with status: {result.status}")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted! Progress saved. Run again to resume.")
                state_manager._save_state()
                # Disconnect before exit
                try:
                    await evaluator.disconnect()
                except:
                    pass
                sys.exit(0)
                
            except Exception as e:
                print(f"✗ Test failed: {str(e)}")
                state_manager.mark_failed(test_id, str(e))
        
        # Save overall results (moved to after statistics so HTML embeds figures/tables)
        
        # Clean disconnect after all tests complete
        # This avoids the asyncio context error
        try:
            await evaluator.disconnect()
        except Exception as e:
            # Silently ignore disconnect errors to avoid the asyncio traceback
            pass
        
        # Get final progress summary
        progress = state_manager.get_progress_summary()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total Tests: {progress['total']}")
        print(f"✅ Completed: {progress['completed']}")
        print(f"❌ Failed: {progress['failed']}")
        print(f"⏭️  Skipped: {progress['skipped']}")
        
        if results:
            successful = sum(1 for r in results if r.status == 'success')
            blocked = sum(1 for r in results if r.status == 'blocked')
            
            print(f"\nOf the newly run tests:")
            print(f"  ✅ Successful: {successful}")
            print(f"  🚫 Bot Blocked: {blocked}")
            
            adapted_count = sum(1 for r in results if r.ux_adaptations and len(r.ux_adaptations) > 0)
            print(f"  🎨 Tests with Adaptations: {adapted_count}")
            
            # Calculate total tokens used
            total_tokens = sum(m.total_tokens for r in results for m in (r.llm_metrics or []))
            print(f"  📈 Total LLM Tokens: {total_tokens:,}")
        
        print(f"\n📄 Results saved to: {output_dir}")
        print("  - .test_state.json: Execution state (for resuming)")
        print("  - overall_results.json: Complete data")
        print("  - overall_results.csv: Summary table")
        print("  - overall_report.html: Visual report")
        print("  - Per-test artifacts in subdirectories")
        
        # Generate comprehensive statistical analysis
        if results:
            print("\n📊 Generating statistical analysis...")
            try:
                if generate_statistics(str(output_dir)):
                    stats_dir = output_dir / "statistical_analysis"
                    print(f"✅ Statistical analysis saved to: {stats_dir}")
                    print("  - aggregated_results.csv: All test results aggregated")
                    print("  - persona_summary.csv: Per-persona statistics")
                    print("  - results_summary.md: Comprehensive report")
                    print("  - delta_distribution.png: Accessibility improvement chart")
                    print("  - wcag_compliance.png: WCAG compliance comparison")
                    if (stats_dir / "models_compare_summary.md").exists():
                        print("  - models_compare_summary.md: Model comparison analysis")
                else:
                    print("⚠️ Statistical analysis generation encountered issues")
            except Exception as e:
                print(f"⚠️ Could not generate statistical analysis: {e}")
            
            # Generate comprehensive analysis with visualizations and dashboard
            if ANALYSIS_AVAILABLE:
                print("\n🔬 Generating comprehensive analysis dashboard...")
                try:
                    analyzer = TestResultsAnalyzer(output_dir)
                    analyzer.run_full_analysis()
                    print(f"✅ Comprehensive analysis saved to: {output_dir / 'analysis_output'}")
                    print("  - figures/: 3 high-quality visualizations")
                    print("  - tables/: CSV and LaTeX tables for publication")
                    print("  - dashboard.html: Interactive results dashboard")
                    print("  - statistical_report.md: Detailed markdown report")
                except Exception as e:
                    print(f"⚠️ Could not generate comprehensive analysis: {e}")
            
            # Generate HTML report AFTER statistics so figures/tables are embedded
            try:
                evaluator.results = results
                evaluator.save_overall_results()
                print("🖼️ Updated overall HTML report with statistical figures and tables.")
            except Exception as e:
                print(f"⚠️ Could not update HTML report with statistics: {e}")
        
        print("\n💡 This evaluation is resumable - if interrupted, just run again!")
        
        return 0
        
    except asyncio.CancelledError:
        print("\n⚠️ Evaluation cancelled by user")
        # Clean disconnect on cancellation
        try:
            await evaluator.disconnect()
        except:
            pass
        return 2
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        # Clean disconnect on interrupt
        try:
            await evaluator.disconnect()
        except:
            pass
        return 2
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Clean disconnect on error
        try:
            await evaluator.disconnect()
        except:
            pass
        return 1


def run():
    """Entry point for the CLI."""
    # Ensure asyncio event loop compatibility on Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()