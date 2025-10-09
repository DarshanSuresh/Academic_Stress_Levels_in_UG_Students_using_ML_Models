# Academic Stress Levels in UG Students using ML Models

## Overview

This comprehensive data science project investigates the impact of meditation practices on academic stress levels in undergraduate students. The project combines multiple analytical approaches:

- **Causal Inference**: Estimate the causal effect of meditation on stress using advanced techniques (Causal Forest, Double Machine Learning)
- **Psychometrics**: Apply Item Response Theory (IRT) to understand latent stress traits and questionnaire item characteristics
- **Machine Learning**: Predict stress levels using classification models and identify student subgroups through clustering
- **Natural Language Processing**: Analyze open-ended survey responses to extract insights about student experiences (optional)

The goal is to provide evidence-based insights that can inform targeted wellness interventions in academic settings.

## Project Structure

```
.
├── data/
│   ├── raw/                    # Raw survey data (CSV files)
│   └── processed/              # Processed and cleaned data
├── notebooks/                  # Jupyter notebooks for analysis and visualization
├── scripts/                    # Python scripts for various analyses
│   ├── data_preprocessing.py   # Data loading, cleaning, and feature engineering
│   ├── causal_inference.py     # Causal effect estimation (Causal Forest, Double ML)
│   ├── irt_analysis.py         # Item Response Theory analysis
│   ├── classification_clustering.py  # ML models for prediction and segmentation
│   └── nlp_analysis.py         # Text analysis of open-ended responses (optional)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License

```

## Key Features

### 1. Data Preprocessing (`scripts/data_preprocessing.py`)
- Automated data loading and cleaning
- Missing value imputation
- Feature engineering (derived variables, composite scores)
- Categorical variable encoding
- Feature scaling and normalization
- Generation of sample data for demonstration

### 2. Causal Inference Analysis (`scripts/causal_inference.py`)
- **Naive ATE**: Baseline treatment effect estimation
- **Causal Forest**: Tree-based causal effect estimation with cross-fitting
- **Double Machine Learning (Double ML)**: Robust causal inference using Neyman-orthogonal approach
- **Conditional ATE (CATE)**: Heterogeneous treatment effect estimation
- Comprehensive visualization and reporting

### 3. Item Response Theory (`scripts/irt_analysis.py`)
- Item statistics and reliability analysis (Cronbach's Alpha)
- Graded Response Model (GRM) for polytomous items
- Person parameter estimation (theta - latent trait)
- Item and test information functions
- Item Characteristic Curves (ICC)
- Person-Item Maps (Wright Maps)

### 4. Classification & Clustering (`scripts/classification_clustering.py`)
- **Classification**: Predict stress levels using multiple algorithms
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machines
- **Clustering**: Identify student subgroups for targeted interventions
  - K-Means clustering
  - Hierarchical clustering
  - Optimal cluster selection (elbow method, silhouette analysis)
- Feature importance analysis
- Model comparison and visualization

### 5. Natural Language Processing (`scripts/nlp_analysis.py`) - Optional
- Text preprocessing and cleaning
- Sentiment analysis of student responses
- Keyword extraction and frequency analysis
- Analysis by stress level and meditation practice
- Visualization of text patterns

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/DarshanSuresh/Academic_Stress_Levels_in_UG_Students_using_ML_Models.git
   cd Academic_Stress_Levels_in_UG_Students_using_ML_Models
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download additional NLP resources (if using NLP module)**
   ```bash
   python -m nltk.downloader punkt stopwords
   python -m spacy download en_core_web_sm
   ```

## Usage

### Quick Start

Each script can be run independently and includes sample data generation for demonstration purposes.

#### 1. Data Preprocessing
```bash
cd scripts
python data_preprocessing.py
```
This will:
- Generate sample survey data (or load from `data/raw/stress_survey_data.csv`)
- Clean and preprocess the data
- Create derived features
- Save processed data to `data/processed/stress_data_processed.csv`

#### 2. Causal Inference Analysis
```bash
python causal_inference.py
```
This will:
- Load processed data
- Estimate treatment effects using multiple methods
- Generate comparison plots and reports
- Save results to `notebooks/`

#### 3. Item Response Theory Analysis
```bash
python irt_analysis.py
```
This will:
- Generate stress questionnaire items
- Calculate reliability statistics
- Fit IRT models
- Create visualizations (ICC, test information, person-item maps)

#### 4. Classification & Clustering
```bash
python classification_clustering.py
```
This will:
- Train multiple classification models
- Perform clustering analysis
- Generate model comparison plots
- Identify student subgroups

#### 5. NLP Analysis (Optional)
```bash
python nlp_analysis.py
```
This will:
- Analyze open-ended text responses
- Perform sentiment analysis
- Extract keywords and themes
- Create text-based visualizations

### Using Jupyter Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Create a new notebook** in the `notebooks/` directory

3. **Import and use the analysis modules**
   ```python
   import sys
   sys.path.append('../scripts')
   
   from data_preprocessing import StressDataPreprocessor
   from causal_inference import CausalInferenceAnalyzer
   from irt_analysis import IRTAnalyzer
   from classification_clustering import StressClassifier, StressClusterer
   from nlp_analysis import TextAnalyzer
   
   # Run your analyses interactively
   ```

### Working with Your Own Data

To use your own survey data:

1. **Prepare your data** as a CSV file with the following recommended columns:
   - `student_id`: Unique identifier
   - `meditation_practice`: Binary (1=practices, 0=does not)
   - Demographics: `age`, `gender`, `year_of_study`
   - Academic: `gpa`, `study_hours_weekly`
   - Outcomes: `stress_score`, `anxiety_score`, `depression_score`
   - Lifestyle: `sleep_hours`, `exercise_frequency`
   - Social: `social_support_score`
   - Optional: `open_ended_response` (text)

2. **Place the file** in `data/raw/stress_survey_data.csv`

3. **Run the preprocessing script** which will automatically detect and use your data

4. **Proceed with analyses** as described in the Quick Start section

## Expected Data Format

The analysis scripts expect data in the following format:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| student_id | int | Unique identifier | 1, 2, 3, ... |
| meditation_practice | int | Treatment indicator | 0 (no) or 1 (yes) |
| age | int | Student age | 18-25 |
| gender | str | Gender | Male, Female, Other |
| year_of_study | int | Academic year | 1-4 |
| gpa | float | Grade point average | 2.0-4.0 |
| study_hours_weekly | int | Study hours per week | 10-60 |
| stress_score | int | Primary outcome (e.g., PSS-10) | 0-40 |
| anxiety_score | int | GAD-7 or similar | 0-21 |
| depression_score | int | PHQ-9 or similar | 0-27 |
| sleep_hours | float | Average sleep hours | 4.0-10.0 |
| exercise_frequency | int | Days per week | 0-7 |
| social_support_score | int | Social support scale | 1-5 |
| open_ended_response | str | Text response (optional) | Free text |

## Results and Outputs

After running the analyses, you'll find:

- **Processed data**: `data/processed/stress_data_processed.csv`
- **Visualizations**: `notebooks/*.png`
  - Causal inference results and comparisons
  - IRT item characteristic curves and information functions
  - Classification confusion matrices and model comparisons
  - Clustering visualizations and elbow curves
  - NLP sentiment analysis and word clouds
- **Console reports**: Detailed statistical summaries and interpretations

## Key Findings and Interpretation

### Causal Inference
- **Negative ATE values**: Meditation *reduces* stress
- **Positive ATE values**: Meditation *increases* stress (unlikely but possible)
- **Values near zero**: No significant effect
- Compare naive vs. adjusted estimates to understand confounding

### IRT Analysis
- **Cronbach's Alpha**: 
  - ≥ 0.9: Excellent reliability
  - 0.8-0.9: Good reliability
  - 0.7-0.8: Acceptable reliability
- **Item Information**: Higher information = better discrimination at that theta level
- **Person-Item Maps**: Shows alignment between person abilities and item difficulties

### Classification
- Model accuracy indicates how well we can predict stress levels
- Feature importance shows which factors are most predictive
- Use for early identification of at-risk students

### Clustering
- Identifies distinct student subgroups (e.g., high-stress/low-support, moderate-stress/active-coping)
- Enables targeted interventions for different groups
- Optimal k determined by elbow method and silhouette scores

## Limitations and Considerations

1. **Causal Inference**: Results based on observational data; unobserved confounders may bias estimates
2. **Sample Size**: Larger samples improve statistical power and reliability
3. **Measurement**: Results depend on the quality and validity of stress measures
4. **Generalizability**: Findings may be specific to the study population
5. **Simplified Implementation**: Some methods (IRT, NLP) use simplified implementations for demonstration

## Future Enhancements

- Integration with more sophisticated IRT packages (e.g., `mirt` via rpy2)
- Advanced NLP with transformer models (BERT, GPT)
- Longitudinal analysis for repeated measures
- Propensity score matching and other causal inference methods
- Interactive dashboards using Plotly Dash or Streamlit
- Automated report generation

## Dependencies

Key packages used in this project:
- **Data Science**: numpy, pandas, scipy, scikit-learn
- **Causal Inference**: econml, causalml, dowhy
- **IRT**: pyirt, py-irt
- **Machine Learning**: xgboost, lightgbm, catboost, hdbscan
- **NLP**: nltk, spacy, transformers, torch
- **Visualization**: matplotlib, seaborn, plotly
- **Statistics**: statsmodels, pingouin

See `requirements.txt` for complete list with versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{stress_meditation_ml,
  author = {Suresh, Darshan},
  title = {Academic Stress Levels in UG Students using ML Models},
  year = {2025},
  url = {https://github.com/DarshanSuresh/Academic_Stress_Levels_in_UG_Students_using_ML_Models}
}
```

## Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

## Acknowledgments

This project draws on established methods from:
- Causal inference literature (Chernozhukov et al., 2018)
- Psychometric theory (Lord, 1980; Embretson & Reise, 2000)
- Machine learning best practices
- Open-source data science community

---

**Note**: This project is for research and educational purposes. Always consult with qualified mental health professionals for actual student wellness interventions.
