# PersonaLayer WebEval

## Profile-Informed Web Personalization Evaluation Tool

This tool is part of the **PersonaLayer** research project, which introduces a novel method for profile-informed web personalization based on 45 user experience (UX) profiles. The tool enables automated evaluation of web interfaces before and after applying persona-specific adaptations.

### Paper Abstract


>
> Static and non-adaptive web interfaces continue to create barriers for people with diverse visual, cognitive, motor, and situational needs. This paper introduces PersonaLayer, a method for profile-informed web personalization based on 45 user experience (UX) profiles developed through AI-assisted exploration, alignment with WCAG 2.2, and expert validation. It enables users to set preferences via a dedicated homepage, toggle adaptations manually, or perform dynamic AI-driven personalization by free-text or voice commands. To test and demonstrate its capability, we provide an AI-driven benchmarking tool that can evaluate before-and-after comparisons and generate automated insights. Initial results show that PersonaLayer can systematically adapt web interfaces across diverse accessibility needs, and the benchmarking tool provides a reproducible way to assess such adaptations without requiring extensive user testing. This work highlights the potential of profile-informed, AI-assisted personalization to reduce accessibility barriers and support more inclusive web experiences.

## Features

- **45 UX Profiles**: Comprehensive profiles covering visual, cognitive, motor, and situational needs
- **Automated Adaptation**: AI-driven CSS and JavaScript generation for each profile
- **Before/After Analysis**: Comprehensive metrics and visual comparison
- **WCAG Compliance Scoring**: Automated assessment of accessibility improvements
- **Bot Detection Handling**: Intelligent detection and bypass strategies
- **Multi-Model Support**: Works with OpenAI, Google, and Anthropic models
- **Detailed Reporting**: HTML reports, JSON data, and CSV summaries

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - Node.js 16+ (for Playwright MCP)
   - OpenRouter API key

2. **Install Python dependencies**:
   ```bash
   pip install httpx dotenv asyncio
   ```

3. **Set up environment**:
   ```bash
   # Create .env file in PL_WebEval directory
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```

4. **Install Playwright MCP** (optional, for browser automation):
   ```bash
   cd src/pl_webeval/playwright-mcp-main
   npm install
   npm run build
   ```

## Usage

### Quick Start

```bash
# Run with default test cases (3 models testing Coupang.com with Low Vision profile)
python run_evaluation.py

# Run with custom test cases
python run_evaluation.py --testcases data/test_cases_sample.csv --output results/experiment1

# Resume interrupted evaluation (automatic)
python run_evaluation.py  # Will resume from last checkpoint if interrupted
```

### Command Line Options

```bash
python run_evaluation.py [OPTIONS]

Options:
  --testcases PATH   Path to test cases CSV file
  --output PATH      Output directory for results
  --api-key KEY      OpenRouter API key (or set OPENROUTER_API_KEY env var)
```

### Test Cases CSV Format

The tool includes default test cases in `data/test_cases_default.csv` that test 3 different models on Coupang.com with the Low Vision profile:

```csv
Website,Task Goal,Task Steps,Success Criteria,LLM,Gen LLM,Ux Profile,TestCaseID,Status
https://www.coupang.com,Evaluate homepage UX and accessibility,Load homepage; analyze baseline; apply UX adaptation; analyze adapted,Page loads and analysis completes,x-ai/grok-4,x-ai/grok-4,Low Vision,TC_47001,
https://www.coupang.com,Evaluate homepage UX and accessibility,Load homepage; analyze baseline; apply UX adaptation; analyze adapted,Page loads and analysis completes,google/gemini-2.5-flash,google/gemini-2.5-flash,Low Vision,TC_47002,
https://www.coupang.com,Evaluate homepage UX and accessibility,Load homepage; analyze baseline; apply UX adaptation; analyze adapted,Page loads and analysis completes,anthropic/claude-sonnet-4,anthropic/claude-sonnet-4,Low Vision,TC_47003,
```

**Columns**:
- `Website`: URL to test
- `Task Goal`: Objective of the test
- `Task Steps`: Steps to perform
- `Success Criteria`: Expected outcome
- `LLM`: Model for evaluation tasks
- `Gen LLM`: Model for generation tasks (CSS/JS)
- `Ux Profile`: Name of UX profile to apply (from profiles CSV)
- `TestCaseID`: Unique test identifier
- `Status`: Leave empty (managed by state tracker, not CSV)

### Supported Models

The tool works with models available through OpenRouter:
- `google/gemini-2.5-flash`
- `anthropic/claude-sonnet-4`
- `x-ai/grok-4`
- And many others available on OpenRouter

**Note**: Test progress is tracked in `.test_state.json` without modifying the input CSV, making tests resumable and version-control friendly.

## UX Profiles

The tool includes 45 predefined UX profiles organized into categories:

**Visual Profiles**:
- Low Vision
- Color Blindness
- Photophobia (Light Sensitivity)
- Dark Environment User

**Cognitive Profiles**:
- High Cognitive Load Sensitivity
- Reading Disorders (e.g., Dyslexia)
- Neurodivergent (e.g., Autism, ADHD)
- Attention or Anxiety Issues

**Motor Profiles**:
- Reduced Dexterity
- Keyboard-Only Navigator
- Touchscreen Optimization

**Situational Profiles**:
- Motion Sensitivity
- Offline-Slow Internet Mode
- Multitasker
- Speed Prioritizer

And many more...

## Output Structure

```
results/
├── run_YYYYMMDD_HHMMSS/
│   ├── overall_results.json       # Complete test data
│   ├── overall_results.csv        # Summary table
│   ├── overall_report.html        # Visual report
│   ├── overall_run_stats.csv      # Statistical summary
│   ├── overall_model_stats.csv    # Per-model metrics
│   ├── overall_profile_stats.csv  # Per-profile analysis
│   └── site_TC_001/              # Per-test artifacts
│       ├── baseline_screenshot.png
│       ├── baseline_snapshot.html
│       ├── adapted_screenshot_Low_Vision.png
│       ├── adapted_snapshot_Low_Vision.html
│       ├── comprehensive_analysis.json
│       └── test_info.txt
```

## Key Metrics

The tool evaluates:

- **Accessibility Score**: 0-100 based on WCAG indicators
- **Visual Complexity**: Layout and hierarchy analysis
- **Text Readability**: Font sizes, spacing, density
- **Color Contrast Issues**: Number of potential problems
- **Adaptation Effectiveness**: How well adaptations work
- **WCAG Compliance**: Levels A, AA, AAA assessment
- **UX Impact**: Usability improvements for target profile

## Research Applications

This tool supports research in:
- Web accessibility evaluation
- Automated personalization effectiveness
- Profile-based adaptation strategies
- WCAG compliance assessment
- Cross-browser/cross-site adaptation patterns
- AI-driven accessibility improvements

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{personalayer2025,
  title={PersonaLayer: Profile-Informed Web Personalization for Enhanced User Experience and Accessibility},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This tool is part of academic research. Please contact the authors for licensing information.

## Contact

For questions or collaboration inquiries, please contact the PersonaLayer research team.

---

**Note**: This tool is designed for research and evaluation purposes. Always validate accessibility improvements with real users from the target populations.
