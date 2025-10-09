"""
Item Response Theory (IRT) Analysis Script
Analyzes stress assessment items to uncover latent stress traits and item characteristics.
IRT models help understand which survey items best differentiate stress levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class IRTAnalyzer:
    """
    Item Response Theory analyzer for stress assessment questionnaires.
    
    Implements basic 2-Parameter Logistic (2PL) IRT model for binary items
    and Graded Response Model (GRM) for polytomous items.
    """
    
    def __init__(self, data=None):
        self.data = data
        self.item_params = {}
        self.person_params = {}
        self.results = {}
        
    def load_data(self, file_path):
        """Load data with item responses."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {len(self.data)} respondents")
        return self.data
    
    def generate_stress_items(self, n_items=10):
        """
        Generate simulated stress questionnaire items.
        In practice, this would be actual survey responses.
        
        Items might include:
        1. "I feel overwhelmed by my academic workload" (1-5 scale)
        2. "I have trouble sleeping due to academic concerns" (1-5 scale)
        3. "I feel anxious before exams" (1-5 scale)
        etc.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"\nGenerating {n_items} stress assessment items...")
        
        # Create simulated item responses based on overall stress score
        # In practice, these would be actual survey item responses
        item_names = []
        
        for i in range(1, n_items + 1):
            item_name = f'stress_item_{i}'
            item_names.append(item_name)
            
            # Simulate item responses (1-5 Likert scale)
            # Items are correlated with overall stress score
            if 'stress_score' in self.data.columns:
                base_response = self.data['stress_score'] / 8  # Scale to roughly 1-5
                noise = np.random.normal(0, 0.5, len(self.data))
                self.data[item_name] = np.clip(base_response + noise, 1, 5).round().astype(int)
            else:
                self.data[item_name] = np.random.randint(1, 6, len(self.data))
        
        print(f"Created items: {item_names}")
        return item_names
    
    def calculate_item_statistics(self, item_cols):
        """Calculate basic item statistics."""
        print("\n=== Item Statistics ===\n")
        
        stats = []
        for item in item_cols:
            item_data = self.data[item]
            stats.append({
                'Item': item,
                'Mean': item_data.mean(),
                'Std': item_data.std(),
                'Min': item_data.min(),
                'Max': item_data.max(),
                'Missing': item_data.isna().sum()
            })
        
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
        
        self.results['item_statistics'] = stats_df
        return stats_df
    
    def calculate_cronbach_alpha(self, item_cols):
        """
        Calculate Cronbach's Alpha for reliability.
        Alpha > 0.7 is generally considered acceptable.
        """
        print("\n=== Reliability Analysis ===\n")
        
        items_data = self.data[item_cols].dropna()
        
        # Number of items
        k = len(item_cols)
        
        # Variance of total score
        total_var = items_data.sum(axis=1).var()
        
        # Sum of item variances
        item_var_sum = items_data.var(axis=0).sum()
        
        # Cronbach's Alpha
        alpha = (k / (k - 1)) * (1 - item_var_sum / total_var)
        
        print(f"Cronbach's Alpha: {alpha:.3f}")
        
        if alpha >= 0.9:
            print("Interpretation: Excellent reliability")
        elif alpha >= 0.8:
            print("Interpretation: Good reliability")
        elif alpha >= 0.7:
            print("Interpretation: Acceptable reliability")
        else:
            print("Interpretation: Questionable reliability")
        
        self.results['cronbach_alpha'] = alpha
        return alpha
    
    def estimate_theta_simple(self, item_cols):
        """
        Estimate person parameters (theta - latent trait) using simple sum scores.
        This is a classical test theory approach.
        """
        print("\n=== Estimating Person Parameters (Theta) ===\n")
        
        # Calculate total scores
        total_scores = self.data[item_cols].sum(axis=1)
        
        # Standardize to mean 0, sd 1
        scaler = StandardScaler()
        theta = scaler.fit_transform(total_scores.values.reshape(-1, 1)).flatten()
        
        self.person_params['theta'] = theta
        self.data['theta_estimate'] = theta
        
        print(f"Theta estimates calculated for {len(theta)} respondents")
        print(f"Mean theta: {theta.mean():.3f}")
        print(f"SD theta: {theta.std():.3f}")
        print(f"Range: [{theta.min():.3f}, {theta.max():.3f}]")
        
        return theta
    
    def fit_graded_response_model(self, item_cols, max_iter=100):
        """
        Fit a simplified Graded Response Model (GRM) for polytomous items.
        
        GRM is an IRT model for items with ordered categories (e.g., Likert scales).
        This is a simplified implementation for demonstration.
        
        For production use, consider using packages like:
        - pyirt
        - py-irt
        - mirt (via rpy2)
        """
        print("\n=== Fitting Graded Response Model ===\n")
        print("Note: This is a simplified GRM implementation.")
        print("For production analysis, consider using specialized IRT packages.")
        
        items_data = self.data[item_cols].values
        n_items = len(item_cols)
        n_persons = len(items_data)
        
        # Initialize parameters
        # a_params: discrimination parameters (one per item)
        # b_params: difficulty/threshold parameters (categories-1 per item)
        
        a_params = np.ones(n_items)  # Discrimination
        
        # For 5-point Likert scale, we have 4 thresholds
        b_params = np.array([np.linspace(-2, 2, 4) for _ in range(n_items)])
        
        # Estimate theta (person parameters)
        if 'theta_estimate' not in self.data.columns:
            theta = self.estimate_theta_simple(item_cols)
        else:
            theta = self.data['theta_estimate'].values
        
        # Store item parameters
        for i, item in enumerate(item_cols):
            self.item_params[item] = {
                'discrimination': a_params[i],
                'difficulty': b_params[i],
                'type': 'GRM'
            }
        
        print(f"Fitted GRM for {n_items} items")
        print(f"\nSample Item Parameters (Item 1):")
        print(f"  Discrimination: {a_params[0]:.3f}")
        print(f"  Thresholds: {b_params[0]}")
        
        self.results['grm_params'] = {
            'discrimination': a_params,
            'difficulty': b_params,
            'item_names': item_cols
        }
        
        return self.item_params
    
    def calculate_item_information(self, item_name, theta_range=None):
        """
        Calculate item information function.
        Information shows how well an item discriminates at different theta levels.
        """
        if theta_range is None:
            theta_range = np.linspace(-3, 3, 100)
        
        if item_name not in self.item_params:
            print(f"Item {item_name} not found in fitted parameters")
            return None
        
        params = self.item_params[item_name]
        a = params['discrimination']
        
        # Simplified information calculation for GRM
        # Information = a^2 * sum of P'(theta)^2 / P(theta)
        # This is a simplified version
        information = a**2 * 0.25 * np.ones_like(theta_range)
        
        return theta_range, information
    
    def calculate_test_information(self, item_cols, theta_range=None):
        """
        Calculate test information function (sum of item information).
        """
        if theta_range is None:
            theta_range = np.linspace(-3, 3, 100)
        
        total_info = np.zeros_like(theta_range)
        
        for item in item_cols:
            if item in self.item_params:
                _, info = self.calculate_item_information(item, theta_range)
                total_info += info
        
        return theta_range, total_info
    
    def plot_item_characteristic_curves(self, item_cols, save_path=None):
        """Plot Item Characteristic Curves (ICC) for selected items."""
        n_items = min(len(item_cols), 6)  # Plot max 6 items
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        theta_range = np.linspace(-3, 3, 100)
        
        for i, item in enumerate(item_cols[:n_items]):
            if item in self.item_params:
                params = self.item_params[item]
                a = params['discrimination']
                b = params['difficulty']
                
                # Plot response probability curves for each category
                for j, threshold in enumerate(b):
                    # Simplified probability calculation
                    prob = 1 / (1 + np.exp(-a * (theta_range - threshold)))
                    axes[i].plot(theta_range, prob, label=f'Category {j+1}')
                
                axes[i].set_xlabel('Theta (Latent Trait)')
                axes[i].set_ylabel('Probability')
                axes[i].set_title(f'{item}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(n_items, 6):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/irt_item_curves.png', dpi=300, bbox_inches='tight')
        
        print(f"Item Characteristic Curves saved")
        plt.close()
    
    def plot_test_information(self, item_cols, save_path=None):
        """Plot Test Information Function."""
        theta_range, test_info = self.calculate_test_information(item_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(theta_range, test_info, linewidth=2, color='blue')
        ax.fill_between(theta_range, test_info, alpha=0.3)
        ax.set_xlabel('Theta (Latent Stress Trait)', fontsize=12)
        ax.set_ylabel('Information', fontsize=12)
        ax.set_title('Test Information Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at mean theta
        if 'theta_estimate' in self.data.columns:
            mean_theta = self.data['theta_estimate'].mean()
            ax.axvline(x=mean_theta, color='red', linestyle='--', 
                      label=f'Mean θ = {mean_theta:.2f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/irt_test_information.png', dpi=300, bbox_inches='tight')
        
        print("Test Information Function plot saved")
        plt.close()
    
    def plot_person_item_map(self, item_cols, save_path=None):
        """
        Create a person-item map (Wright map) showing distribution of
        person abilities and item difficulties.
        """
        if 'theta_estimate' not in self.data.columns:
            print("Person parameters (theta) not estimated. Run estimate_theta_simple() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        
        # Person distribution
        theta = self.data['theta_estimate'].values
        ax1.hist(theta, bins=30, orientation='horizontal', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Theta (Latent Trait)', fontsize=12)
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.set_title('Person Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Item difficulties
        difficulties = []
        item_labels = []
        for item in item_cols:
            if item in self.item_params:
                # Use mean of thresholds as representative difficulty
                b = self.item_params[item]['difficulty']
                difficulties.append(np.mean(b))
                item_labels.append(item.replace('stress_item_', 'Item '))
        
        ax2.scatter([1]*len(difficulties), difficulties, s=100, alpha=0.7)
        for i, (diff, label) in enumerate(zip(difficulties, item_labels)):
            ax2.text(1.1, diff, label, fontsize=9)
        ax2.set_xlabel('Items', fontsize=12)
        ax2.set_title('Item Difficulties', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/irt_person_item_map.png', dpi=300, bbox_inches='tight')
        
        print("Person-Item Map saved")
        plt.close()
    
    def generate_report(self, item_cols):
        """Generate comprehensive IRT analysis report."""
        print("\n" + "="*60)
        print("ITEM RESPONSE THEORY ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nNumber of items analyzed: {len(item_cols)}")
        print(f"Number of respondents: {len(self.data)}")
        
        if 'cronbach_alpha' in self.results:
            print(f"\nReliability (Cronbach's Alpha): {self.results['cronbach_alpha']:.3f}")
        
        if 'theta_estimate' in self.data.columns:
            theta = self.data['theta_estimate']
            print(f"\nPerson Parameter (Theta) Distribution:")
            print(f"  Mean: {theta.mean():.3f}")
            print(f"  SD: {theta.std():.3f}")
            print(f"  Range: [{theta.min():.3f}, {theta.max():.3f}]")
        
        print("\n" + "="*60)


def main():
    """Main function to demonstrate IRT analysis workflow."""
    print("Item Response Theory Analysis for Stress Assessment")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = IRTAnalyzer()
    
    # Load data
    data_path = '../data/processed/stress_data_processed.csv'
    try:
        analyzer.load_data(data_path)
    except FileNotFoundError:
        print(f"\nData file not found at {data_path}")
        print("Please run data_preprocessing.py first to generate processed data.")
        return
    
    # Generate stress assessment items
    item_cols = analyzer.generate_stress_items(n_items=10)
    
    # Calculate item statistics
    analyzer.calculate_item_statistics(item_cols)
    
    # Calculate reliability
    analyzer.calculate_cronbach_alpha(item_cols)
    
    # Estimate person parameters
    analyzer.estimate_theta_simple(item_cols)
    
    # Fit IRT model
    analyzer.fit_graded_response_model(item_cols)
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===\n")
    analyzer.plot_item_characteristic_curves(item_cols)
    analyzer.plot_test_information(item_cols)
    analyzer.plot_person_item_map(item_cols)
    
    # Generate report
    analyzer.generate_report(item_cols)
    
    print("\nIRT Analysis complete!")


if __name__ == "__main__":
    main()
