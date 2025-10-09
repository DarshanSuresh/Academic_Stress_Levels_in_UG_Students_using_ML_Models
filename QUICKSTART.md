# Quick Start Guide

This guide will help you get started with the Academic Stress Analysis project in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 5-10 minutes of setup time

## Setup (3 steps)

### 1. Clone and Navigate
```bash
git clone https://github.com/DarshanSuresh/Academic_Stress_Levels_in_UG_Students_using_ML_Models.git
cd Academic_Stress_Levels_in_UG_Students_using_ML_Models
```

### 2. Install Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 3. Validate Installation
```bash
python validate_project.py
```

If you see "✓ ALL VALIDATION CHECKS PASSED!" you're ready to go!

## Running Your First Analysis (30 seconds)

### Option A: Run Individual Scripts

```bash
cd scripts

# 1. Preprocess data (generates sample data automatically)
python data_preprocessing.py

# 2. Run causal inference analysis
python causal_inference.py

# 3. Run IRT analysis
python irt_analysis.py

# 4. Run classification & clustering
python classification_clustering.py

# 5. Run NLP analysis (optional)
python nlp_analysis.py
```

### Option B: Use Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/example_workflow.ipynb
# Run all cells to see the complete analysis
```

## What You'll Get

After running the analyses, you'll have:

1. **Processed Data**: `data/processed/stress_data_processed.csv`
2. **Visualizations**: Multiple PNG files in `notebooks/` showing:
   - Causal effect estimates
   - IRT curves and information functions
   - Model performance comparisons
   - Cluster visualizations
   - Sentiment analysis results
3. **Console Reports**: Detailed statistical summaries and interpretations

## Next Steps

- **Use Your Own Data**: Place your CSV file in `data/raw/stress_survey_data.csv`
- **Customize Analysis**: Modify the scripts to suit your research questions
- **Create Reports**: Use the Jupyter notebook to generate comprehensive reports
- **Explore Results**: Review visualizations and statistical outputs

## Common Issues

### Issue: "ModuleNotFoundError"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: "No data file found"
**Solution**: This is normal! Scripts automatically generate sample data for demonstration.

### Issue: Scripts run but no visualizations appear
**Solution**: Check the `notebooks/` directory for saved PNG files.

## Getting Help

- Review the full [README.md](README.md) for detailed documentation
- Check individual script docstrings for module-specific help
- Open an issue on GitHub for bugs or questions

## Quick Reference

| Analysis | Script | Output |
|----------|--------|--------|
| Data Prep | `data_preprocessing.py` | Cleaned dataset |
| Causal Effects | `causal_inference.py` | Treatment effect estimates |
| Psychometrics | `irt_analysis.py` | Item parameters, reliability |
| Prediction | `classification_clustering.py` | Model accuracy, clusters |
| Text Analysis | `nlp_analysis.py` | Sentiment, keywords |

## Estimated Run Times

- Data preprocessing: ~10 seconds
- Causal inference: ~30 seconds
- IRT analysis: ~20 seconds
- Classification & clustering: ~40 seconds
- NLP analysis: ~15 seconds
- **Total**: ~2 minutes for all analyses

---

**Ready to dive deeper?** Check out the [full documentation](README.md) for advanced usage and customization options.
