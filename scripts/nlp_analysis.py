"""
Natural Language Processing (NLP) Analysis Script (Optional)
Analyzes open-ended text responses about stress and meditation experiences
using sentiment analysis, topic modeling, and text mining techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TextAnalyzer:
    """
    NLP analyzer for open-ended survey responses.
    
    This is a simplified implementation using basic NLP techniques.
    For advanced analysis, consider using spaCy, transformers, or BERT models.
    """
    
    def __init__(self, data=None):
        self.data = data
        self.results = {}
        
    def load_data(self, file_path):
        """Load data with text responses."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {len(self.data)} rows")
        return self.data
    
    def generate_sample_responses(self, n_samples=None):
        """
        Generate sample open-ended responses for demonstration.
        In practice, these would be actual survey responses.
        """
        if n_samples is None:
            n_samples = len(self.data) if self.data is not None else 100
        
        # Sample responses about stress and meditation
        positive_responses = [
            "Meditation has really helped me manage my academic stress.",
            "I feel more focused and calm after meditating regularly.",
            "Practicing mindfulness before exams reduces my anxiety significantly.",
            "I sleep better and feel less overwhelmed with meditation.",
            "Meditation gives me tools to cope with academic pressure.",
            "I'm more productive and less stressed since starting meditation.",
            "Deep breathing exercises help me stay composed during finals.",
            "Meditation has improved my mental health and academic performance.",
        ]
        
        negative_responses = [
            "I find it hard to make time for meditation with my busy schedule.",
            "Academic workload is overwhelming regardless of meditation.",
            "Exams and assignments cause constant stress and worry.",
            "I struggle to concentrate and feel anxious most of the time.",
            "The pressure to maintain good grades is exhausting.",
            "I don't see much benefit from meditation practices.",
            "Stress from multiple deadlines is affecting my health.",
        ]
        
        neutral_responses = [
            "I sometimes practice meditation when I remember to.",
            "Academic stress varies depending on the semester.",
            "I'm trying different stress management techniques.",
            "My stress levels change based on assignment deadlines.",
            "I attend wellness workshops offered by the university.",
            "Managing time between studies and self-care is challenging.",
        ]
        
        # Randomly assign responses based on stress levels
        if self.data is not None and 'stress_score' in self.data.columns:
            responses = []
            for score in self.data['stress_score'].head(n_samples):
                if score < 20:
                    responses.append(np.random.choice(positive_responses))
                elif score > 30:
                    responses.append(np.random.choice(negative_responses))
                else:
                    responses.append(np.random.choice(neutral_responses))
            
            self.data['open_ended_response'] = responses[:len(self.data)]
        else:
            # Generate random mix
            all_responses = positive_responses + negative_responses + neutral_responses
            responses = np.random.choice(all_responses, size=n_samples)
            if self.data is not None:
                self.data['open_ended_response'] = responses[:len(self.data)]
        
        print(f"Generated {n_samples} sample responses")
        return self.data
    
    def basic_text_preprocessing(self, text_col='open_ended_response'):
        """
        Basic text preprocessing without external NLP libraries.
        For production, use nltk or spaCy for better preprocessing.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column {text_col} not found")
        
        print("\n=== Text Preprocessing ===\n")
        
        # Convert to lowercase
        self.data['text_clean'] = self.data[text_col].str.lower()
        
        # Remove punctuation (simple approach)
        self.data['text_clean'] = self.data['text_clean'].str.replace('[^a-zA-Z0-9\\s]', '', regex=True)
        
        # Remove extra whitespace
        self.data['text_clean'] = self.data['text_clean'].str.strip()
        
        print(f"Preprocessed {len(self.data)} text responses")
        
        return self.data['text_clean']
    
    def calculate_text_length(self, text_col='text_clean'):
        """Calculate text length statistics."""
        self.data['text_length'] = self.data[text_col].str.split().str.len()
        
        print("\n=== Text Length Statistics ===\n")
        print(f"Mean length: {self.data['text_length'].mean():.1f} words")
        print(f"Median length: {self.data['text_length'].median():.1f} words")
        print(f"Range: {self.data['text_length'].min()}-{self.data['text_length'].max()} words")
        
        return self.data['text_length']
    
    def simple_sentiment_analysis(self, text_col='text_clean'):
        """
        Simple sentiment analysis using keyword matching.
        For production, use VADER, TextBlob, or transformer models.
        """
        print("\n=== Sentiment Analysis ===\n")
        
        # Define sentiment keywords
        positive_words = {'help', 'calm', 'better', 'improved', 'beneficial', 
                         'positive', 'good', 'great', 'excellent', 'reduces',
                         'focused', 'productive', 'cope', 'manage'}
        
        negative_words = {'stress', 'anxiety', 'overwhelm', 'pressure', 'worry',
                         'exhausting', 'struggle', 'difficult', 'hard', 'bad',
                         'terrible', 'anxious', 'depressed', 'overwhelming'}
        
        sentiments = []
        sentiment_scores = []
        
        for text in self.data[text_col]:
            words = set(str(text).split())
            
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            
            score = pos_count - neg_count
            sentiment_scores.append(score)
            
            if score > 0:
                sentiments.append('Positive')
            elif score < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        
        self.data['sentiment'] = sentiments
        self.data['sentiment_score'] = sentiment_scores
        
        print("Sentiment distribution:")
        print(self.data['sentiment'].value_counts().to_string())
        
        self.results['sentiment'] = {
            'distribution': self.data['sentiment'].value_counts(),
            'mean_score': np.mean(sentiment_scores)
        }
        
        return sentiments, sentiment_scores
    
    def extract_keywords(self, text_col='text_clean', top_n=20):
        """Extract most common keywords."""
        print(f"\n=== Top {top_n} Keywords ===\n")
        
        # Common stop words to exclude
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its'}
        
        # Extract all words
        all_words = []
        for text in self.data[text_col].dropna():
            words = str(text).split()
            all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
        
        # Count frequencies
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_n)
        
        print("Word : Frequency")
        for word, freq in top_words:
            print(f"{word:15} : {freq}")
        
        self.results['keywords'] = top_words
        
        return top_words
    
    def analyze_by_stress_level(self, text_col='text_clean', stress_col='stress_score'):
        """Analyze text patterns by stress level."""
        print("\n=== Text Analysis by Stress Level ===\n")
        
        if stress_col not in self.data.columns:
            print(f"Column {stress_col} not found")
            return
        
        # Create stress groups
        self.data['stress_group'] = pd.qcut(
            self.data[stress_col], q=3, 
            labels=['Low Stress', 'Medium Stress', 'High Stress']
        )
        
        # Analyze by group
        for group in ['Low Stress', 'Medium Stress', 'High Stress']:
            group_data = self.data[self.data['stress_group'] == group]
            
            if 'sentiment' in self.data.columns:
                sentiment_dist = group_data['sentiment'].value_counts()
                print(f"\n{group} (n={len(group_data)}):")
                print(f"  Sentiment: {sentiment_dist.to_dict()}")
            
            if 'text_length' in self.data.columns:
                avg_length = group_data['text_length'].mean()
                print(f"  Avg response length: {avg_length:.1f} words")
    
    def analyze_meditation_effects(self, text_col='text_clean', treatment_col='meditation_practice'):
        """Analyze text patterns by meditation practice."""
        print("\n=== Text Analysis by Meditation Practice ===\n")
        
        if treatment_col not in self.data.columns:
            print(f"Column {treatment_col} not found")
            return
        
        for group in [0, 1]:
            group_name = "Practices Meditation" if group == 1 else "No Meditation"
            group_data = self.data[self.data[treatment_col] == group]
            
            print(f"\n{group_name} (n={len(group_data)}):")
            
            if 'sentiment' in self.data.columns:
                sentiment_dist = group_data['sentiment'].value_counts()
                print(f"  Sentiment: {sentiment_dist.to_dict()}")
            
            if 'sentiment_score' in self.data.columns:
                avg_score = group_data['sentiment_score'].mean()
                print(f"  Avg sentiment score: {avg_score:.2f}")
    
    def plot_sentiment_analysis(self, save_path=None):
        """Visualize sentiment analysis results."""
        if 'sentiment' not in self.data.columns:
            print("Sentiment analysis not performed. Run simple_sentiment_analysis() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Sentiment distribution
        sentiment_counts = self.data['sentiment'].value_counts()
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, alpha=0.7, 
                   color=['green', 'gray', 'red'])
        axes[0].set_xlabel('Sentiment', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Sentiment by meditation practice
        if 'meditation_practice' in self.data.columns:
            sentiment_by_meditation = pd.crosstab(
                self.data['meditation_practice'], 
                self.data['sentiment'],
                normalize='index'
            ) * 100
            
            sentiment_by_meditation.plot(kind='bar', ax=axes[1], alpha=0.7)
            axes[1].set_xlabel('Meditation Practice', fontsize=12)
            axes[1].set_ylabel('Percentage (%)', fontsize=12)
            axes[1].set_title('Sentiment by Meditation Practice', fontsize=14, fontweight='bold')
            axes[1].set_xticklabels(['No', 'Yes'], rotation=0)
            axes[1].legend(title='Sentiment')
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/nlp_sentiment_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        print("Sentiment analysis plot saved")
        plt.close()
    
    def plot_word_frequency(self, top_n=15, save_path=None):
        """Plot word frequency visualization."""
        if 'keywords' not in self.results:
            print("Keyword extraction not performed. Run extract_keywords() first.")
            return
        
        keywords = self.results['keywords'][:top_n]
        words = [w[0] for w in keywords]
        freqs = [w[1] for w in keywords]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(words, freqs, alpha=0.7, color='steelblue')
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Keywords', fontsize=12)
        ax.set_title(f'Top {top_n} Most Frequent Keywords', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/nlp_word_frequency.png', 
                       dpi=300, bbox_inches='tight')
        
        print("Word frequency plot saved")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive NLP analysis report."""
        print("\n" + "="*60)
        print("NLP ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nTotal responses analyzed: {len(self.data)}")
        
        if 'text_length' in self.data.columns:
            print(f"\nAverage response length: {self.data['text_length'].mean():.1f} words")
        
        if 'sentiment' in self.data.columns:
            print("\nSentiment Distribution:")
            for sentiment, count in self.data['sentiment'].value_counts().items():
                pct = 100 * count / len(self.data)
                print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        if 'keywords' in self.results:
            print("\nTop 10 Keywords:")
            for i, (word, freq) in enumerate(self.results['keywords'][:10], 1):
                print(f"  {i}. {word} ({freq} occurrences)")
        
        print("\n" + "="*60)


def main():
    """Main function to demonstrate NLP analysis workflow."""
    print("NLP Analysis for Open-Ended Stress Survey Responses")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Load data
    data_path = '../data/processed/stress_data_processed.csv'
    try:
        analyzer.load_data(data_path)
    except FileNotFoundError:
        print(f"\nData file not found at {data_path}")
        print("Please run data_preprocessing.py first.")
        return
    
    # Generate sample responses (in practice, use actual survey responses)
    analyzer.generate_sample_responses()
    
    # Text preprocessing
    analyzer.basic_text_preprocessing()
    
    # Calculate text length
    analyzer.calculate_text_length()
    
    # Sentiment analysis
    analyzer.simple_sentiment_analysis()
    
    # Extract keywords
    analyzer.extract_keywords(top_n=20)
    
    # Analyze by stress level
    analyzer.analyze_by_stress_level()
    
    # Analyze by meditation practice
    analyzer.analyze_meditation_effects()
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===\n")
    analyzer.plot_sentiment_analysis()
    analyzer.plot_word_frequency(top_n=15)
    
    # Generate report
    analyzer.generate_report()
    
    print("\nNLP Analysis complete!")
    print("\nNote: This is a simplified NLP implementation.")
    print("For production analysis, consider using:")
    print("  - NLTK or spaCy for advanced text processing")
    print("  - VADER or TextBlob for sentiment analysis")
    print("  - Transformers (BERT, RoBERTa) for deep learning approaches")
    print("  - LDA or BERTopic for topic modeling")


if __name__ == "__main__":
    main()
