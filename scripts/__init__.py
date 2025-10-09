"""
Academic Stress Analysis Scripts
=================================

This package contains analysis modules for studying meditation's impact 
on academic stress in undergraduate students.

Modules:
--------
- data_preprocessing: Data loading, cleaning, and feature engineering
- causal_inference: Causal effect estimation (Causal Forest, Double ML)
- irt_analysis: Item Response Theory analysis
- classification_clustering: ML models for prediction and segmentation
- nlp_analysis: Text analysis of open-ended responses

Usage:
------
    from data_preprocessing import StressDataPreprocessor
    from causal_inference import CausalInferenceAnalyzer
    from irt_analysis import IRTAnalyzer
    from classification_clustering import StressClassifier, StressClusterer
    from nlp_analysis import TextAnalyzer
"""

__version__ = "1.0.0"
__author__ = "Darshan Suresh"

__all__ = [
    'StressDataPreprocessor',
    'CausalInferenceAnalyzer',
    'IRTAnalyzer',
    'StressClassifier',
    'StressClusterer',
    'TextAnalyzer'
]
