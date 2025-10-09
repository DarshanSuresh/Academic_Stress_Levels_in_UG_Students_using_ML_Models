# Notebooks Directory

This directory contains Jupyter notebooks for interactive analysis and visualization.

## Getting Started

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Create a new notebook** or use the example provided

3. **Import analysis modules**
   ```python
   import sys
   sys.path.append('../scripts')
   
   from data_preprocessing import StressDataPreprocessor
   from causal_inference import CausalInferenceAnalyzer
   from irt_analysis import IRTAnalyzer
   from classification_clustering import StressClassifier, StressClusterer
   from nlp_analysis import TextAnalyzer
   ```

## Suggested Notebooks

Create separate notebooks for each analysis component:

- `01_data_exploration.ipynb` - Initial data exploration and visualization
- `02_preprocessing.ipynb` - Interactive data preprocessing
- `03_causal_inference.ipynb` - Causal effect estimation
- `04_irt_analysis.ipynb` - Item Response Theory modeling
- `05_classification.ipynb` - Stress prediction models
- `06_clustering.ipynb` - Student subgroup identification
- `07_nlp_analysis.ipynb` - Text analysis of responses
- `08_comprehensive_report.ipynb` - Final integrated analysis

## Visualization Outputs

Analysis scripts automatically save plots to this directory:
- `causal_inference_results.png`
- `irt_item_curves.png`
- `irt_test_information.png`
- `irt_person_item_map.png`
- `classification_confusion_matrices.png`
- `classification_model_comparison.png`
- `clustering_elbow_curve.png`
- `clustering_kmeans_pca.png`
- `nlp_sentiment_analysis.png`
- `nlp_word_frequency.png`

## Tips

- Save your notebooks regularly
- Use markdown cells to document your analysis
- Include visualizations inline for better reporting
- Clear outputs before committing to version control (to reduce file size)
