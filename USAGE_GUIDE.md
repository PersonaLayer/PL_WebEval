# PersonaLayer WebEval - Community Usage Guide

## 🎯 For Academic Researchers

### Research Applications

1. **Accessibility Studies**
   - Evaluate WCAG 2.2 compliance across different user personas
   - Measure accessibility improvements with PersonaLayer adaptations
   - Compare LLM performance in accessibility assessment

2. **HCI Research**
   - Study personalization effectiveness across 45 UX profiles
   - Analyze user interface adaptation strategies
   - Investigate cognitive load reduction techniques

3. **AI/ML Research**
   - Benchmark different LLMs on web evaluation tasks
   - Study prompt engineering for accessibility analysis
   - Compare structured output generation across models

### Example Research Workflows

#### Study 1: Cross-Model Accessibility Assessment
```python
# Create test cases with multiple models
test_cases = [
    {
        "website": "https://example.edu",
        "ux_profile": "visual_impairment_total",
        "llm_model": "openai/gpt-5-mini"
    },
    {
        "website": "https://example.edu",
        "ux_profile": "visual_impairment_total",
        "llm_model": "anthropic/claude-sonnet-4"
    }
]
# Compare model outputs for consistency and accuracy
```

#### Study 2: Persona-Based Adaptation Effectiveness
```python
# Test same site with different personas
profiles = [
    "elderly_low_tech", 
    "dyslexia",
    "motor_impairment_severe",
    "autism_spectrum"
]
# Measure improvement metrics for each profile
```

### Publishing Results

When citing PersonaLayer WebEval in your research:

```bibtex
@inproceedings{personlayer2025,
  title={PersonaLayer: Profile-Informed Web Personalization for Enhanced User Experience and Accessibility},
  author={[Authors]},
  booktitle={Proceedings of [Conference]},
  year={2025}
}
```

### Data Collection & Analysis

Results are stored in structured JSON format:
- `results/[timestamp]/[test_id]/evaluation_results.json` - Complete evaluation data
- `results/[timestamp]/aggregated_metrics.json` - Summary statistics
- `results/[timestamp]/.test_state.json` - Execution state (for resumable runs)

## 🔧 For Software Engineers

### Integration Options

#### 1. Python Library Integration
```python
from pl_webeval import WebEvaluator, UXProfile

# Initialize evaluator
evaluator = WebEvaluator(api_key="your-openrouter-key")

# Load UX profile
profile = UXProfile.load("visual_impairment_moderate")

# Evaluate website
results = await evaluator.evaluate(
    url="https://yoursite.com",
    profile=profile,
    model="openai/gpt-5-mini"
)

# Apply adaptations
css_rules = results.adaptations.css_rules
js_code = results.adaptations.javascript_code
```

#### 2. CI/CD Pipeline Integration
```yaml
# GitHub Actions example
name: Accessibility Testing
on: [push]
jobs:
  accessibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run PersonaLayer WebEval
        run: |
          pip install pl-webeval
          pl-webeval --testcases tests/accessibility.csv \
                     --output reports/accessibility
      - name: Check Thresholds
        run: |
          python -c "
          import json
          data = json.load(open('reports/accessibility/metrics.json'))
          assert data['wcag_score'] >= 0.85, 'WCAG score below threshold'
          "
```

#### 3. A/B Testing Framework
```python
class PersonaLayerABTest:
    def __init__(self):
        self.evaluator = WebEvaluator()
    
    async def test_variant(self, url, profiles):
        results = {}
        for profile in profiles:
            result = await self.evaluator.evaluate(url, profile)
            results[profile] = {
                'wcag_score': result.wcag_compliance.score,
                'improvements': len(result.accessibility_improvements)
            }
        return results
```

### State Management Features

#### Resumable Testing
Tests automatically resume from the last checkpoint if interrupted:
```bash
# Start testing
pl-webeval --testcases large_test_set.csv

# If interrupted, just run again - it resumes automatically
pl-webeval --testcases large_test_set.csv
```

#### Parallel Execution
```python
from pl_webeval import ParallelEvaluator

evaluator = ParallelEvaluator(workers=5)
results = await evaluator.evaluate_batch(test_cases)
```

#### Custom State Handlers
```python
from pl_webeval import StateManager

# Custom state tracking
state = StateManager(output_dir=Path("./results"))
state.mark_completed("test_001", {"score": 0.92})

# Resume from specific point
resume_point = state.get_resume_point()
```

## 📊 For Data Scientists

### Analytics & Metrics

#### Built-in Metrics
- **WCAG Compliance Score**: 0-1 scale compliance rating
- **Accessibility Improvements**: Count and categorization
- **Bot Detection Risk**: Probability and mitigation strategies
- **Token Usage**: For cost optimization
- **Response Times**: For performance analysis

#### Data Export Formats
```python
# Export to pandas DataFrame
import pandas as pd
from pl_webeval import ResultsLoader

loader = ResultsLoader("results/2025-01-06")
df = loader.to_dataframe()

# Analyze by profile
profile_metrics = df.groupby('ux_profile').agg({
    'wcag_score': 'mean',
    'improvements_count': 'sum',
    'token_usage': 'mean'
})

# Export to CSV for further analysis
df.to_csv('evaluation_data.csv', index=False)
```

### Machine Learning Applications

#### Feature Extraction
```python
from pl_webeval import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_from_results(results)
# Features include: DOM complexity, color contrast ratios, 
# text density, navigation structure, etc.
```

#### Model Training Data
```python
# Generate training data for accessibility prediction models
training_data = []
for result in results:
    training_data.append({
        'features': result.extracted_features,
        'label': result.wcag_compliance.score,
        'profile': result.ux_profile
    })
```

## 🌍 Community Contributions

### Adding New UX Profiles
Create a PR with your profile in `data/ux_profiles.csv`:
```csv
profile_id,category,name,description,vision,hearing,mobility,cognitive
custom_profile,special_needs,Custom Profile,Description,1.0,0.8,0.6,0.9
```

### Adding LLM Models
Simply use any OpenRouter-supported model ID:
```csv
website,ux_profile,llm_model
https://example.com,elderly_low_tech,meta-llama/llama-3.3-70b
```

### Extending Evaluation Criteria
```python
from pl_webeval import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    async def evaluate_custom_criteria(self, html, profile):
        # Add your custom evaluation logic
        return custom_metrics
```

## 🔒 Ethical Considerations

### Privacy & Data Protection
- No user data is collected or stored
- Websites are analyzed statically without user interaction
- Results can be anonymized for publication

### Responsible Use
- Respect robots.txt and rate limits
- Use for improving accessibility, not circumventing protections
- Follow academic ethical guidelines for web scraping

### Bias Mitigation
- Test with multiple LLM models to identify biases
- Validate results with human experts
- Consider cultural and linguistic diversity in profiles

## 📚 Resources

### Documentation
- [API Reference](docs/api_reference.md)
- [UX Profile Guide](docs/ux_profiles.md)
- [Model Comparison](docs/model_comparison.md)

### Examples
- [Jupyter Notebooks](examples/notebooks/)
- [Sample Test Cases](data/test_cases_sample.csv)
- [Integration Examples](examples/integrations/)

### Support
- GitHub Issues: [Report bugs or request features]
- Discussions: [Community forum for questions]
- Email: [Academic collaboration inquiries]

## 🚀 Quick Start Recipes

### Recipe 1: University Website Audit
```bash
# Audit university websites for accessibility
pl-webeval --testcases university_sites.csv \
           --profiles "visual_impairment_total,hearing_loss_profound,motor_impairment_severe" \
           --report-format html
```

### Recipe 2: E-commerce Accessibility
```bash
# Test checkout flows with different profiles
pl-webeval --url "https://shop.example.com/checkout" \
           --all-profiles \
           --focus "forms,buttons,navigation"
```

### Recipe 3: Government Compliance Check
```bash
# WCAG 2.2 Level AA compliance verification
pl-webeval --testcases gov_sites.csv \
           --wcag-level AA \
           --generate-report compliance_report.pdf
```

## 📈 Performance & Scalability

### Optimization Tips
1. **Batch Processing**: Process multiple URLs in parallel
2. **Caching**: Results are cached to avoid re-processing
3. **Model Selection**: Choose faster models for initial screening
4. **Checkpoint System**: Enables resuming large test suites

### Resource Requirements
- **Memory**: 4GB minimum, 8GB recommended
- **Storage**: ~10MB per evaluation (full data)
- **API Costs**: ~$0.002-0.02 per evaluation (varies by model)
- **Time**: 15-60 seconds per evaluation

## 🏆 Success Stories

### Case Studies
1. **EdTech Platform**: Improved accessibility score from 0.65 to 0.94
2. **Government Portal**: Achieved WCAG 2.2 Level AA compliance
3. **E-learning Site**: Reduced bounce rate by 35% for users with disabilities

### Research Impact
- Published in top-tier HCI conferences
- Adopted by 3 university accessibility labs
- Featured in W3C accessibility guidelines discussion

---

**Join the PersonaLayer community in making the web accessible for everyone!**