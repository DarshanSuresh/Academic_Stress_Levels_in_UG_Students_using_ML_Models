"""
Classification and Clustering Analysis Script
Predicts stress levels (classification) and identifies student subgroups (clustering)
for targeted wellness interventions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, roc_curve)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StressClassifier:
    """
    Classifier for predicting stress levels in students.
    """
    
    def __init__(self, data=None):
        self.data = data
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def load_data(self, file_path):
        """Load preprocessed data."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {len(self.data)} rows")
        return self.data
    
    def create_stress_categories(self, outcome_col='stress_score', method='tertiles'):
        """
        Create categorical stress levels from continuous stress scores.
        
        Parameters:
        -----------
        outcome_col : str
            Name of the continuous stress score column
        method : str
            'tertiles' - split into low/medium/high
            'binary' - split into low/high at median
            'quartiles' - split into 4 groups
        """
        if outcome_col not in self.data.columns:
            raise ValueError(f"Column {outcome_col} not found in data")
        
        if method == 'tertiles':
            self.data['stress_category'] = pd.qcut(
                self.data[outcome_col], q=3, 
                labels=['Low', 'Medium', 'High']
            )
            print("\nStress categories created (tertiles):")
        elif method == 'binary':
            median = self.data[outcome_col].median()
            self.data['stress_category'] = (self.data[outcome_col] > median).map(
                {False: 'Low', True: 'High'}
            )
            print("\nStress categories created (binary):")
        elif method == 'quartiles':
            self.data['stress_category'] = pd.qcut(
                self.data[outcome_col], q=4,
                labels=['Very Low', 'Low', 'High', 'Very High']
            )
            print("\nStress categories created (quartiles):")
        
        print(self.data['stress_category'].value_counts().to_string())
        
        return self.data['stress_category']
    
    def prepare_features(self, target_col='stress_category', exclude_cols=None):
        """
        Prepare features and target for classification.
        """
        if target_col not in self.data.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        if exclude_cols is None:
            exclude_cols = ['student_id', 'stress_score', target_col, 'open_ended_response']
        
        # Select feature columns
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols and 
                       self.data[col].dtype in [np.int64, np.float64]]
        
        X = self.data[feature_cols].values
        y = self.data[target_col].values
        
        print(f"\nFeatures prepared:")
        print(f"  Number of features: {len(feature_cols)}")
        print(f"  Number of samples: {len(X)}")
        print(f"  Target classes: {np.unique(y)}")
        
        return X, y, feature_cols
    
    def train_models(self, X, y):
        """
        Train multiple classification models.
        """
        print("\n=== Training Classification Models ===\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print()
        
        # Identify best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = best_model_name
        
        print(f"Best Model: {best_model_name} (Accuracy: {self.results[best_model_name]['accuracy']:.3f})")
        
        return self.models, self.results
    
    def get_feature_importance(self, feature_names, top_n=10):
        """
        Get feature importance from the best tree-based model.
        """
        if self.best_model is None:
            print("No models trained yet.")
            return None
        
        model = self.results[self.best_model]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\n=== Top {top_n} Feature Importances ({self.best_model}) ===\n")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            return importances, indices
        else:
            print(f"{self.best_model} does not support feature importance.")
            return None, None
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 4:
                break
                
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, square=True)
            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/classification_confusion_matrices.png', 
                       dpi=300, bbox_inches='tight')
        
        print("Confusion matrices plot saved")
        plt.close()
    
    def plot_model_comparison(self, save_path=None):
        """Plot comparison of model performances."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        ax.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        ax.errorbar(x + width/2, cv_means, yerr=cv_stds, fmt='none', 
                   color='black', capsize=5)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/classification_model_comparison.png', 
                       dpi=300, bbox_inches='tight')
        
        print("Model comparison plot saved")
        plt.close()


class StressClusterer:
    """
    Clustering analyzer to identify student subgroups.
    """
    
    def __init__(self, data=None):
        self.data = data
        self.clusters = {}
        self.results = {}
        
    def load_data(self, file_path):
        """Load preprocessed data."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {len(self.data)} rows")
        return self.data
    
    def prepare_features(self, exclude_cols=None, scale=True):
        """Prepare features for clustering."""
        if exclude_cols is None:
            exclude_cols = ['student_id', 'open_ended_response', 'stress_category']
        
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols and 
                       self.data[col].dtype in [np.int64, np.float64]]
        
        X = self.data[feature_cols].values
        
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            print("Features scaled")
        
        print(f"\nFeatures prepared for clustering:")
        print(f"  Number of features: {len(feature_cols)}")
        print(f"  Number of samples: {len(X)}")
        
        return X, feature_cols
    
    def find_optimal_k(self, X, k_range=range(2, 11)):
        """Find optimal number of clusters using elbow method."""
        print("\n=== Finding Optimal Number of Clusters ===\n")
        
        inertias = []
        silhouette_scores = []
        
        from sklearn.metrics import silhouette_score
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.3f}")
        
        self.results['elbow'] = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
        return inertias, silhouette_scores
    
    def fit_kmeans(self, X, n_clusters=3):
        """Fit K-Means clustering."""
        print(f"\n=== K-Means Clustering (k={n_clusters}) ===\n")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.clusters['kmeans'] = labels
        self.data['cluster_kmeans'] = labels
        
        print(f"Cluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} students ({100*count/len(labels):.1f}%)")
        
        self.results['kmeans'] = {
            'model': kmeans,
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
        
        return labels
    
    def fit_hierarchical(self, X, n_clusters=3):
        """Fit Hierarchical clustering."""
        print(f"\n=== Hierarchical Clustering (k={n_clusters}) ===\n")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X)
        
        self.clusters['hierarchical'] = labels
        self.data['cluster_hierarchical'] = labels
        
        print(f"Cluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} students ({100*count/len(labels):.1f}%)")
        
        self.results['hierarchical'] = {
            'model': hierarchical,
            'labels': labels
        }
        
        return labels
    
    def characterize_clusters(self, method='kmeans'):
        """Characterize clusters by their features."""
        if method not in self.clusters:
            print(f"Clustering method '{method}' not found.")
            return
        
        cluster_col = f'cluster_{method}'
        
        print(f"\n=== Cluster Characterization ({method}) ===\n")
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in [cluster_col]]
        
        # Calculate mean for each cluster
        cluster_profiles = self.data.groupby(cluster_col)[numeric_cols].mean()
        
        print("Cluster Profiles (mean values):")
        print(cluster_profiles.to_string())
        
        # Focus on key variables
        key_vars = ['stress_score', 'meditation_practice', 'gpa', 
                   'anxiety_score', 'depression_score']
        available_vars = [v for v in key_vars if v in numeric_cols]
        
        if available_vars:
            print(f"\nKey Variables by Cluster:")
            print(cluster_profiles[available_vars].to_string())
        
        self.results[f'{method}_profiles'] = cluster_profiles
        
        return cluster_profiles
    
    def plot_elbow_curve(self, save_path=None):
        """Plot elbow curve for optimal k selection."""
        if 'elbow' not in self.results:
            print("Elbow analysis not performed. Run find_optimal_k() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        k_range = self.results['elbow']['k_range']
        inertias = self.results['elbow']['inertias']
        silhouette = self.results['elbow']['silhouette_scores']
        
        # Elbow plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_range, silhouette, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../notebooks/clustering_elbow_curve.png', 
                       dpi=300, bbox_inches='tight')
        
        print("Elbow curve plot saved")
        plt.close()
    
    def plot_clusters_pca(self, X, method='kmeans', save_path=None):
        """Visualize clusters using PCA."""
        if method not in self.clusters:
            print(f"Clustering method '{method}' not found.")
            return
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        labels = self.clusters[method]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                           cmap='viridis', alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                     fontsize=12)
        ax.set_title(f'Cluster Visualization ({method.title()})', 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, label='Cluster', ax=ax)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'../notebooks/clustering_{method}_pca.png', 
                       dpi=300, bbox_inches='tight')
        
        print(f"Cluster visualization ({method}) saved")
        plt.close()


def main():
    """Main function to demonstrate classification and clustering workflow."""
    print("Classification and Clustering Analysis for Academic Stress")
    print("=" * 60)
    
    data_path = '../data/processed/stress_data_processed.csv'
    
    # CLASSIFICATION
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS")
    print("="*60)
    
    classifier = StressClassifier()
    
    try:
        classifier.load_data(data_path)
    except FileNotFoundError:
        print(f"\nData file not found at {data_path}")
        print("Please run data_preprocessing.py first.")
        return
    
    # Create stress categories
    classifier.create_stress_categories(method='tertiles')
    
    # Prepare features
    X, y, feature_names = classifier.prepare_features()
    
    # Train models
    classifier.train_models(X, y)
    
    # Feature importance
    classifier.get_feature_importance(feature_names, top_n=10)
    
    # Visualizations
    classifier.plot_confusion_matrices()
    classifier.plot_model_comparison()
    
    # CLUSTERING
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    clusterer = StressClusterer()
    clusterer.load_data(data_path)
    
    # Prepare features
    X_cluster, cluster_features = clusterer.prepare_features(scale=True)
    
    # Find optimal k
    clusterer.find_optimal_k(X_cluster)
    
    # Fit clustering models
    clusterer.fit_kmeans(X_cluster, n_clusters=3)
    clusterer.fit_hierarchical(X_cluster, n_clusters=3)
    
    # Characterize clusters
    clusterer.characterize_clusters(method='kmeans')
    
    # Visualizations
    clusterer.plot_elbow_curve()
    clusterer.plot_clusters_pca(X_cluster, method='kmeans')
    
    print("\nClassification and Clustering analysis complete!")


if __name__ == "__main__":
    main()
