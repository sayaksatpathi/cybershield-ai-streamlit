import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """
    Comprehensive fraud detection model training and evaluation system.
    Supports multiple algorithms and handles class imbalance.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the fraud detection model trainer."""
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str], 
                    target_column: str = 'is_fraud', test_size: float = 0.2) -> Tuple:
        """Prepare data for training with proper train/test split."""
        print("Preparing data for training...")
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Separate features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"Fraud rate in training: {y_train.mean():.3f}")
        print(f"Fraud rate in test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self) -> Dict[str, Any]:
        """Define the models to train and compare."""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                verbose=-1
            ),
            
            'svm': SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        return models
    
    def create_pipelines(self, models: Dict[str, Any], use_smote: bool = True) -> Dict[str, Pipeline]:
        """Create ML pipelines with preprocessing and sampling."""
        pipelines = {}
        
        for name, model in models.items():
            if use_smote and name != 'svm':  # SMOTE can be memory intensive with SVM
                # Pipeline with SMOTE for handling class imbalance
                pipeline = ImbPipeline([
                    ('scaler', StandardScaler()),
                    ('smote', SMOTE(random_state=self.random_state)),
                    ('classifier', model)
                ])
            else:
                # Simple pipeline with just scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
            
            pipelines[name] = pipeline
        
        return pipelines
    
    def evaluate_model(self, pipeline: Pipeline, X_train: pd.DataFrame, 
                      X_test: pd.DataFrame, y_train: pd.Series, 
                      y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a single model and return metrics."""
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': pipeline.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def cross_validate_models(self, pipelines: Dict[str, Pipeline], 
                            X_train: pd.DataFrame, y_train: pd.Series, 
                            cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation for all models."""
        print("Performing cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, pipeline in pipelines.items():
            print(f"Cross-validating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=skf, scoring='roc_auc', n_jobs=-1)
            
            cv_results[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
        
        return cv_results
    
    def train_and_evaluate_all(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, 
                             use_smote: bool = True) -> Dict[str, Any]:
        """Train and evaluate all models."""
        print("Training and evaluating all models...")
        
        # Define models
        models = self.define_models()
        
        # Create pipelines
        pipelines = self.create_pipelines(models, use_smote)
        
        # Store pipelines
        self.models = pipelines
        
        # Cross-validation
        cv_results = self.cross_validate_models(pipelines, X_train, y_train)
        
        # Train and evaluate each model
        results = {}
        
        for name, pipeline in pipelines.items():
            print(f"Training and evaluating {name}...")
            
            metrics, y_pred, y_pred_proba = self.evaluate_model(
                pipeline, X_train, X_test, y_train, y_test
            )
            
            results[name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'cv_results': cv_results[name],
                'pipeline': pipeline
            }
        
        # Store results
        self.results = results
        
        # Find best model based on F1 score
        best_f1 = 0
        for name, result in results.items():
            if result['metrics']['f1_score'] > best_f1:
                best_f1 = result['metrics']['f1_score']
                self.best_model_name = name
                self.best_model = result['pipeline']
        
        print(f"Best model: {self.best_model_name} (F1 Score: {best_f1:.3f})")
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of all model results."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create summary dataframe
        summary_data = []
        for name, result in results.items():
            metrics = result['metrics']
            cv_results = result['cv_results']
            
            summary_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1 Score': f"{metrics['f1_score']:.3f}",
                'ROC AUC': f"{metrics['roc_auc']:.3f}",
                'Avg Precision': f"{metrics['avg_precision']:.3f}",
                'CV AUC': f"{cv_results['cv_auc_mean']:.3f} ± {cv_results['cv_auc_std']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\nBest Model: {self.best_model_name}")
    
    def plot_model_comparison(self, results: Dict[str, Any], save_path: str = None):
        """Plot comparison of model performance."""
        # Extract metrics for plotting
        models = list(results.keys())
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [results[model]['metrics'][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Highlight best model
            best_idx = np.argmax(values)
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(1.0)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{values[j]:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, results: Dict[str, Any], y_test: pd.Series, save_path: str = None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            y_pred_proba = result['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = result['metrics']['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, results: Dict[str, Any], y_test: pd.Series, 
                                   save_path: str = None):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            y_pred_proba = result['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = result['metrics']['avg_precision']
            
            plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add baseline (random classifier)
        baseline = y_test.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Random Classifier (AP = {baseline:.3f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str, y_test: pd.Series, save_path: str = None):
        """Plot confusion matrix for a specific model."""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return
        
        y_pred = self.results[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'], 
                   yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from the best model or specified model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return pd.DataFrame()
        
        pipeline = self.results[model_name]['pipeline']
        
        # Get the classifier from the pipeline
        if hasattr(pipeline, 'named_steps'):
            classifier = pipeline.named_steps['classifier']
        else:
            classifier = pipeline.steps[-1][1]  # Last step should be classifier
        
        # Extract feature importance
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importance = np.abs(classifier.coef_[0])
        else:
            print(f"Model {model_name} doesn't support feature importance.")
            return pd.DataFrame()
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 20, 
                              save_path: str = None):
        """Plot feature importance for the specified model."""
        importance_df = self.get_feature_importance(model_name)
        
        if importance_df.empty:
            return
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name or self.best_model_name}')
        plt.gca().invert_yaxis()
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return importance_df
    
    def save_best_model(self, filepath: str):
        """Save the best model to disk."""
        if self.best_model is None:
            print("No best model found. Train models first.")
            return
        
        # Save the model
        joblib.dump(self.best_model, filepath)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'metrics': self.results[self.best_model_name]['metrics']
        }
        
        metadata_filepath = filepath.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_filepath)
        
        print(f"Best model saved to {filepath}")
        print(f"Model metadata saved to {metadata_filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model from disk."""
        # Load the model
        self.best_model = joblib.load(filepath)
        
        # Load metadata
        metadata_filepath = filepath.replace('.pkl', '_metadata.pkl')
        try:
            metadata = joblib.load(metadata_filepath)
            self.best_model_name = metadata['model_name']
            self.feature_columns = metadata['feature_columns']
            print(f"Model {self.best_model_name} loaded successfully.")
            print(f"Features: {len(self.feature_columns)}")
        except FileNotFoundError:
            print("Model loaded but metadata not found.")
    
    def predict_transaction(self, transaction_features: Dict[str, Any]) -> Tuple[int, float]:
        """Predict if a single transaction is fraudulent."""
        if self.best_model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([transaction_features])
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select only the required features in the correct order
        X = df[self.feature_columns]
        
        # Make prediction
        prediction = self.best_model.predict(X)[0]
        probability = self.best_model.predict_proba(X)[0, 1]
        
        return prediction, probability

if __name__ == "__main__":
    # Load engineered features
    print("Loading engineered features...")
    df = pd.read_csv('transaction_features.csv')
    
    # Load feature list
    with open('feature_list.txt', 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    
    # Initialize model trainer
    model_trainer = FraudDetectionModel(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(
        df, feature_columns, target_column='is_fraud', test_size=0.2
    )
    
    # Train and evaluate all models
    results = model_trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test, use_smote=True
    )
    
    # Print results summary
    model_trainer.print_results_summary(results)
    
    # Create visualizations
    model_trainer.plot_model_comparison(results, 'model_comparison.png')
    model_trainer.plot_roc_curves(results, y_test, 'roc_curves.png')
    model_trainer.plot_precision_recall_curves(results, y_test, 'precision_recall_curves.png')
    model_trainer.plot_confusion_matrix(model_trainer.best_model_name, y_test, 'confusion_matrix.png')
    
    # Plot feature importance
    importance_df = model_trainer.plot_feature_importance(
        model_name=model_trainer.best_model_name, 
        top_n=20, 
        save_path='feature_importance.png'
    )
    
    # Save the best model
    model_trainer.save_best_model('fraud_detection_model.pkl')
    
    print("\nTraining completed! All results and plots have been saved.")
