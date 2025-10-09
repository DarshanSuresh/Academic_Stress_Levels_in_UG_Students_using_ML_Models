"""
Data Preprocessing Script for Academic Stress Study
This script handles data loading, cleaning, and preprocessing for studying
meditation's impact on undergraduate students' academic stress.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')


class StressDataPreprocessor:
    """
    Preprocessor for academic stress survey data.
    
    Expected data columns:
    - student_id: Unique identifier
    - meditation_practice: Binary (1=practices meditation, 0=does not)
    - age, gender, year_of_study: Demographics
    - gpa, study_hours_weekly: Academic metrics
    - stress_score: Outcome variable (e.g., PSS-10 scale)
    - anxiety_score, depression_score: Mental health measures
    - sleep_hours, exercise_frequency: Lifestyle factors
    - social_support_score: Social factors
    - open_ended_response: Optional text responses
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path=None):
        """Load data from CSV file."""
        path = file_path or self.data_path
        if path is None:
            raise ValueError("Data path must be provided")
            
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Generating sample data...")
            self.data = self._generate_sample_data()
        else:
            self.data = pd.read_csv(path)
            print(f"Loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
        
        return self.data
    
    def _generate_sample_data(self, n_samples=500):
        """Generate sample data for demonstration purposes."""
        np.random.seed(42)
        
        data = {
            'student_id': range(1, n_samples + 1),
            'meditation_practice': np.random.binomial(1, 0.3, n_samples),
            'age': np.random.randint(18, 25, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'year_of_study': np.random.randint(1, 5, n_samples),
            'gpa': np.random.uniform(2.0, 4.0, n_samples),
            'study_hours_weekly': np.random.randint(10, 60, n_samples),
            'stress_score': np.random.randint(10, 40, n_samples),
            'anxiety_score': np.random.randint(0, 21, n_samples),
            'depression_score': np.random.randint(0, 27, n_samples),
            'sleep_hours': np.random.uniform(4, 10, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'social_support_score': np.random.randint(1, 5, n_samples),
        }
        
        # Add treatment effect: meditation reduces stress
        meditation_effect = data['meditation_practice'] * np.random.uniform(-5, -2, n_samples)
        data['stress_score'] = data['stress_score'] + meditation_effect
        data['stress_score'] = np.clip(data['stress_score'], 0, 40).astype(int)
        
        df = pd.DataFrame(data)
        print(f"Generated sample data with {len(df)} rows")
        return df
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values in the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Identify numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy=strategy)
            self.data[numeric_cols] = imputer_numeric.fit_transform(self.data[numeric_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = imputer_categorical.fit_transform(self.data[categorical_cols])
        
        print(f"Missing values handled using '{strategy}' strategy")
        return self.data
    
    def encode_categorical_variables(self):
        """Encode categorical variables using Label Encoding."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['student_id', 'open_ended_response']:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
        
        return self.data
    
    def create_feature_groups(self):
        """Create feature groups for different types of variables."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        feature_groups = {
            'treatment': ['meditation_practice'],
            'demographics': ['age', 'gender', 'year_of_study'],
            'academic': ['gpa', 'study_hours_weekly'],
            'mental_health': ['anxiety_score', 'depression_score'],
            'lifestyle': ['sleep_hours', 'exercise_frequency'],
            'social': ['social_support_score'],
            'outcome': ['stress_score']
        }
        
        # Filter to only include columns that exist in the data
        available_groups = {}
        for group_name, cols in feature_groups.items():
            available_cols = [col for col in cols if col in self.data.columns]
            if available_cols:
                available_groups[group_name] = available_cols
        
        print("\nFeature groups created:")
        for group_name, cols in available_groups.items():
            print(f"  {group_name}: {cols}")
        
        return available_groups
    
    def scale_features(self, exclude_cols=None):
        """Scale numeric features using StandardScaler."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if exclude_cols is None:
            exclude_cols = ['student_id', 'meditation_practice', 'stress_score']
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if cols_to_scale:
            self.data[cols_to_scale] = self.scaler.fit_transform(self.data[cols_to_scale])
            print(f"Scaled {len(cols_to_scale)} numeric features")
        
        return self.data
    
    def create_derived_features(self):
        """Create derived features for analysis."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Academic pressure index
        if 'gpa' in self.data.columns and 'study_hours_weekly' in self.data.columns:
            self.data['academic_pressure'] = (
                self.data['study_hours_weekly'] / self.data['gpa']
            )
        
        # Mental health composite
        if 'anxiety_score' in self.data.columns and 'depression_score' in self.data.columns:
            self.data['mental_health_composite'] = (
                self.data['anxiety_score'] + self.data['depression_score']
            )
        
        # Wellness index
        if 'sleep_hours' in self.data.columns and 'exercise_frequency' in self.data.columns:
            self.data['wellness_index'] = (
                self.data['sleep_hours'] * 0.5 + self.data['exercise_frequency'] * 0.5
            )
        
        print("Derived features created")
        return self.data
    
    def preprocess_pipeline(self, scale=True, create_derived=True):
        """Run the full preprocessing pipeline."""
        print("\n=== Starting Preprocessing Pipeline ===\n")
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Create derived features
        if create_derived:
            self.create_derived_features()
        
        # Scale features
        if scale:
            self.scale_features()
        
        # Create feature groups
        feature_groups = self.create_feature_groups()
        
        self.processed_data = self.data.copy()
        
        print("\n=== Preprocessing Complete ===\n")
        print(f"Processed data shape: {self.processed_data.shape}")
        
        return self.processed_data, feature_groups
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV file."""
        if self.processed_data is None:
            raise ValueError("No processed data. Run preprocess_pipeline() first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    def get_summary_statistics(self):
        """Generate summary statistics for the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n=== Summary Statistics ===\n")
        print(self.data.describe())
        
        if 'meditation_practice' in self.data.columns:
            print(f"\nMeditation practitioners: {self.data['meditation_practice'].sum()}")
            print(f"Non-practitioners: {len(self.data) - self.data['meditation_practice'].sum()}")
        
        return self.data.describe()


def main():
    """Main function to demonstrate preprocessing workflow."""
    print("Academic Stress Data Preprocessing")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = StressDataPreprocessor()
    
    # Load data (will generate sample data if file doesn't exist)
    data_path = '../data/raw/stress_survey_data.csv'
    preprocessor.load_data(data_path)
    
    # Get summary statistics
    preprocessor.get_summary_statistics()
    
    # Run preprocessing pipeline
    processed_data, feature_groups = preprocessor.preprocess_pipeline(
        scale=True,
        create_derived=True
    )
    
    # Save processed data
    output_path = '../data/processed/stress_data_processed.csv'
    preprocessor.save_processed_data(output_path)
    
    print("\nPreprocessing complete!")
    print(f"Processed data available at: {output_path}")
    
    return processed_data, feature_groups


if __name__ == "__main__":
    processed_data, feature_groups = main()
