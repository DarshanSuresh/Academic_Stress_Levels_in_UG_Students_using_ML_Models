"""
Causal Inference Analysis Script
Implements Causal Forest and Double Machine Learning (Double ML) to estimate
the causal effect of meditation on academic stress in undergraduate students.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class CausalInferenceAnalyzer:
    """
    Causal inference analyzer for treatment effect estimation.
    Implements Causal Forest and Double ML approaches.
    """
    
    def __init__(self, data=None):
        self.data = data
        self.results = {}
        
    def load_data(self, file_path):
        """Load preprocessed data."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {len(self.data)} rows")
        return self.data
    
    def prepare_data_for_causal_inference(self, treatment_col='meditation_practice',
                                         outcome_col='stress_score',
                                         exclude_cols=None):
        """
        Prepare data for causal inference analysis.
        
        Parameters:
        -----------
        treatment_col : str
            Name of the treatment variable
        outcome_col : str
            Name of the outcome variable
        exclude_cols : list
            Columns to exclude from confounders
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if exclude_cols is None:
            exclude_cols = ['student_id', treatment_col, outcome_col]
        
        # Extract treatment
        T = self.data[treatment_col].values
        
        # Extract outcome
        Y = self.data[outcome_col].values
        
        # Extract confounders (all other numeric columns)
        covariate_cols = [col for col in self.data.columns 
                         if col not in exclude_cols and 
                         self.data[col].dtype in [np.int64, np.float64]]
        X = self.data[covariate_cols].values
        
        print(f"\nData prepared for causal inference:")
        print(f"  Treatment: {treatment_col}")
        print(f"  Outcome: {outcome_col}")
        print(f"  Confounders: {len(covariate_cols)} variables")
        print(f"  Sample size: {len(T)}")
        print(f"  Treated units: {T.sum()} ({100*T.mean():.1f}%)")
        
        return T, Y, X, covariate_cols
    
    def estimate_ate_naive(self, T, Y):
        """
        Estimate naive Average Treatment Effect (ATE) without adjustment.
        This serves as a baseline comparison.
        """
        treated_outcomes = Y[T == 1]
        control_outcomes = Y[T == 0]
        
        ate_naive = treated_outcomes.mean() - control_outcomes.mean()
        
        # Calculate standard error
        se = np.sqrt(treated_outcomes.var() / len(treated_outcomes) + 
                    control_outcomes.var() / len(control_outcomes))
        
        print(f"\n=== Naive ATE (without adjustment) ===")
        print(f"ATE: {ate_naive:.3f}")
        print(f"Standard Error: {se:.3f}")
        print(f"95% CI: [{ate_naive - 1.96*se:.3f}, {ate_naive + 1.96*se:.3f}]")
        
        self.results['naive_ate'] = {
            'estimate': ate_naive,
            'se': se,
            'ci_lower': ate_naive - 1.96*se,
            'ci_upper': ate_naive + 1.96*se
        }
        
        return ate_naive
    
    def causal_forest_ate(self, T, Y, X, n_estimators=100, random_state=42):
        """
        Estimate Average Treatment Effect using a simplified Causal Forest approach.
        
        Note: This is a simplified implementation. For production use, consider
        using econml.dml.CausalForestDML or grf package.
        """
        print("\n=== Causal Forest Estimation ===")
        
        # Split data for cross-fitting
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, test_size=0.5, random_state=random_state
        )
        
        # Step 1: Model the outcome
        y_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        y_model.fit(X_train, Y_train)
        Y_pred = y_model.predict(X_test)
        Y_residual = Y_test - Y_pred
        
        # Step 2: Model the treatment
        t_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        t_model.fit(X_train, T_train)
        T_pred = t_model.predict(X_test)
        T_residual = T_test - T_pred
        
        # Step 3: Estimate treatment effect using residuals
        # This is the Robinson transformation / Partial Linear Model approach
        valid_idx = np.abs(T_residual) > 1e-10
        ate_cf = np.sum(Y_residual[valid_idx] * T_residual[valid_idx]) / np.sum(T_residual[valid_idx]**2)
        
        # Bootstrap for confidence intervals
        n_bootstrap = 100
        ate_bootstrap = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y_residual), size=len(Y_residual), replace=True)
            valid = np.abs(T_residual[idx]) > 1e-10
            if valid.sum() > 0:
                ate_boot = np.sum(Y_residual[idx][valid] * T_residual[idx][valid]) / np.sum(T_residual[idx][valid]**2)
                ate_bootstrap.append(ate_boot)
        
        ate_se = np.std(ate_bootstrap)
        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)
        
        print(f"Causal Forest ATE: {ate_cf:.3f}")
        print(f"Standard Error: {ate_se:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        self.results['causal_forest'] = {
            'estimate': ate_cf,
            'se': ate_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'y_model': y_model,
            't_model': t_model
        }
        
        return ate_cf
    
    def double_ml_ate(self, T, Y, X, n_folds=5, random_state=42):
        """
        Estimate Average Treatment Effect using Double Machine Learning (Double ML).
        
        This implements the Neyman-orthogonal approach with cross-fitting.
        Reference: Chernozhukov et al. (2018)
        """
        print("\n=== Double Machine Learning Estimation ===")
        
        from sklearn.model_selection import KFold
        
        n_samples = len(Y)
        theta_estimates = []
        
        # Cross-fitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Stage 1: Predict Y using X
            y_learner = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            y_learner.fit(X_train, Y_train)
            Y_pred = y_learner.predict(X_test)
            Y_residual = Y_test - Y_pred
            
            # Stage 2: Predict T using X
            t_learner = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            t_learner.fit(X_train, T_train)
            T_pred = t_learner.predict(X_test)
            T_residual = T_test - T_pred
            
            # Stage 3: Estimate treatment effect
            valid_idx = np.abs(T_residual) > 1e-10
            if valid_idx.sum() > 0:
                theta_fold = np.sum(Y_residual[valid_idx] * T_residual[valid_idx]) / np.sum(T_residual[valid_idx]**2)
                theta_estimates.append(theta_fold)
        
        # Aggregate estimates
        ate_dml = np.mean(theta_estimates)
        ate_se = np.std(theta_estimates) / np.sqrt(n_folds)
        ci_lower = ate_dml - 1.96 * ate_se
        ci_upper = ate_dml + 1.96 * ate_se
        
        print(f"Double ML ATE: {ate_dml:.3f}")
        print(f"Standard Error: {ate_se:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        self.results['double_ml'] = {
            'estimate': ate_dml,
            'se': ate_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fold_estimates': theta_estimates
        }
        
        return ate_dml
    
    def estimate_cate(self, T, Y, X, feature_names=None):
        """
        Estimate Conditional Average Treatment Effects (CATE).
        This shows how treatment effects vary across different subgroups.
        """
        print("\n=== Conditional Average Treatment Effect (CATE) ===")
        
        # Train separate models for treated and control
        treated_idx = T == 1
        control_idx = T == 0
        
        # Model for treated
        model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_treated.fit(X[treated_idx], Y[treated_idx])
        
        # Model for control
        model_control = RandomForestRegressor(n_estimators=100, random_state=42)
        model_control.fit(X[control_idx], Y[control_idx])
        
        # Predict counterfactuals
        mu1 = model_treated.predict(X)  # Potential outcome under treatment
        mu0 = model_control.predict(X)  # Potential outcome under control
        
        # CATE is the difference
        cate = mu1 - mu0
        
        print(f"CATE mean: {cate.mean():.3f}")
        print(f"CATE std: {cate.std():.3f}")
        print(f"CATE range: [{cate.min():.3f}, {cate.max():.3f}]")
        
        self.results['cate'] = {
            'estimates': cate,
            'mean': cate.mean(),
            'std': cate.std(),
            'model_treated': model_treated,
            'model_control': model_control
        }
        
        return cate
    
    def plot_results(self, save_path=None):
        """Plot comparison of different causal inference methods."""
        if not self.results:
            print("No results to plot. Run analysis methods first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: ATE Comparison
        methods = []
        estimates = []
        ci_lower = []
        ci_upper = []
        
        for method in ['naive_ate', 'causal_forest', 'double_ml']:
            if method in self.results:
                methods.append(method.replace('_', ' ').title())
                estimates.append(self.results[method]['estimate'])
                ci_lower.append(self.results[method]['ci_lower'])
                ci_upper.append(self.results[method]['ci_upper'])
        
        y_pos = np.arange(len(methods))
        axes[0].barh(y_pos, estimates, alpha=0.7)
        axes[0].errorbar(estimates, y_pos, 
                        xerr=[(est - ci_lower[i], ci_upper[i] - est) 
                              for i, est in enumerate(estimates)],
                        fmt='none', color='black', capsize=5)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(methods)
        axes[0].set_xlabel('Average Treatment Effect (ATE)')
        axes[0].set_title('Comparison of ATE Estimates')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: CATE Distribution
        if 'cate' in self.results:
            cate = self.results['cate']['estimates']
            axes[1].hist(cate, bins=30, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=cate.mean(), color='red', linestyle='--', 
                          label=f'Mean: {cate.mean():.2f}')
            axes[1].set_xlabel('Conditional Average Treatment Effect (CATE)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Individual Treatment Effects')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig('../notebooks/causal_inference_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved to: ../notebooks/causal_inference_results.png")
        
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive report of causal inference results."""
        print("\n" + "="*60)
        print("CAUSAL INFERENCE ANALYSIS REPORT")
        print("="*60)
        
        print("\nResearch Question:")
        print("What is the causal effect of meditation practice on academic stress?")
        
        print("\n--- RESULTS SUMMARY ---\n")
        
        for method, result in self.results.items():
            if 'estimate' in result:
                print(f"{method.replace('_', ' ').title()}:")
                print(f"  Estimate: {result['estimate']:.3f}")
                print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
                print()
        
        print("\nInterpretation:")
        print("- Negative values indicate meditation REDUCES stress")
        print("- Positive values indicate meditation INCREASES stress")
        print("- Values close to 0 indicate no effect")
        
        print("\nNote: These results are based on observational data.")
        print("Causal Forest and Double ML methods adjust for confounding,")
        print("but unobserved confounders may still bias estimates.")


def main():
    """Main function to demonstrate causal inference workflow."""
    print("Causal Inference Analysis for Academic Stress Study")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CausalInferenceAnalyzer()
    
    # Load data
    data_path = '../data/processed/stress_data_processed.csv'
    try:
        analyzer.load_data(data_path)
    except FileNotFoundError:
        print(f"\nData file not found at {data_path}")
        print("Please run data_preprocessing.py first to generate processed data.")
        return
    
    # Prepare data
    T, Y, X, feature_names = analyzer.prepare_data_for_causal_inference(
        treatment_col='meditation_practice',
        outcome_col='stress_score'
    )
    
    # Estimate effects
    print("\n" + "="*60)
    print("ESTIMATING CAUSAL EFFECTS")
    print("="*60)
    
    # Naive ATE
    analyzer.estimate_ate_naive(T, Y)
    
    # Causal Forest
    analyzer.causal_forest_ate(T, Y, X)
    
    # Double ML
    analyzer.double_ml_ate(T, Y, X)
    
    # CATE
    analyzer.estimate_cate(T, Y, X, feature_names)
    
    # Generate visualizations
    analyzer.plot_results()
    
    # Generate report
    analyzer.generate_report()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
