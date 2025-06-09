import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import openai
import os

class EvaluatorAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def evaluate_model(self, model_results: dict, cleaned_state: dict) -> dict:
        """Comprehensive model evaluation and reporting"""
        evaluation_results = {
            "performance_summary": {},
            "visualizations": [],
            "detailed_metrics": {},
            "final_report": {},
            "recommendations": []
        }
        
        try:
            print("📊 Generating performance summary...")
            evaluation_results["performance_summary"] = self._create_performance_summary(model_results)
            
            print("📈 Creating evaluation visualizations...")
            evaluation_results["visualizations"] = self._create_evaluation_plots(model_results, cleaned_state)
            
            print("🔍 Computing detailed metrics...")
            evaluation_results["detailed_metrics"] = self._compute_detailed_metrics(model_results)
            
            print("📋 Generating final report...")
            evaluation_results["final_report"] = self._generate_final_report(model_results, evaluation_results)
            
            print("💡 Creating recommendations...")
            evaluation_results["recommendations"] = self._generate_recommendations(model_results, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def _create_performance_summary(self, model_results: dict) -> dict:
        """Create a summary of model performance"""
        metrics = model_results["training_metrics"]
        problem_type = model_results["problem_type"]
        
        if problem_type == "classification":
            return {
                "model_type": model_results["model_type"],
                "problem_type": problem_type,
                "accuracy": metrics["test_accuracy"],
                "performance_tier": model_results.get("ai_insights", {}).get("performance_tier", "Unknown"),
                "features_count": len(model_results["features"]),
                "target_variable": model_results["target"]
            }
        else:
            return {
                "model_type": model_results["model_type"],
                "problem_type": problem_type,
                "r2_score": metrics["test_r2"],
                "rmse": metrics["test_rmse"],
                "performance_tier": model_results.get("ai_insights", {}).get("performance_tier", "Unknown"),
                "features_count": len(model_results["features"]),
                "target_variable": model_results["target"]
            }
    
    def _create_evaluation_plots(self, model_results: dict, cleaned_state: dict) -> list:
        """Create evaluation visualizations"""
        visualizations = []
        
        try:
            plt.style.use('seaborn-v0_8')
            
            # Performance metrics visualization
            metrics = model_results["training_metrics"]
            problem_type = model_results["problem_type"]
            
            if problem_type == "classification":
                # Classification metrics plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Accuracy comparison
                train_acc = metrics["train_accuracy"]
                test_acc = metrics["test_accuracy"]
                
                ax1.bar(['Training', 'Testing'], [train_acc, test_acc], 
                       color=['skyblue', 'lightcoral'])
                ax1.set_title('Model Accuracy Comparison')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                
                # Add value labels on bars
                for i, v in enumerate([train_acc, test_acc]):
                    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
                # Feature importance (if available)
                if hasattr(model_results["model"], 'feature_importances_'):
                    importances = model_results["model"].feature_importances_
                    features = model_results["features"][:10]  # Top 10 features
                    importances = importances[:10]
                    
                    ax2.barh(features, importances, color='lightgreen')
                    ax2.set_title('Top 10 Feature Importances')
                    ax2.set_xlabel('Importance')
                else:
                    ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Feature Importance')
                
                plt.tight_layout()
                plt.savefig('classification_evaluation.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("classification_evaluation.png")
                
            else:
                # Regression metrics plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # R² and RMSE comparison
                train_r2 = metrics["train_r2"]
                test_r2 = metrics["test_r2"]
                train_rmse = metrics["train_rmse"]
                test_rmse = metrics["test_rmse"]
                
                # R² scores
                ax1.bar(['Training', 'Testing'], [train_r2, test_r2], 
                       color=['skyblue', 'lightcoral'])
                ax1.set_title('R² Score Comparison')
                ax1.set_ylabel('R² Score')
                ax1.set_ylim(0, 1)
                
                for i, v in enumerate([train_r2, test_r2]):
                    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
                # RMSE scores
                ax2.bar(['Training', 'Testing'], [train_rmse, test_rmse], 
                       color=['lightgreen', 'orange'])
                ax2.set_title('RMSE Comparison')
                ax2.set_ylabel('RMSE')
                
                for i, v in enumerate([train_rmse, test_rmse]):
                    ax2.text(i, v + (max(train_rmse, test_rmse) * 0.01), f'{v:.3f}', 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("regression_evaluation.png")
            
            return visualizations
            
        except Exception as e:
            print(f"⚠️ Warning: Evaluation plot creation failed: {str(e)}")
            return []
    
    def _compute_detailed_metrics(self, model_results: dict) -> dict:
        """Compute additional detailed metrics"""
        metrics = model_results["training_metrics"]
        problem_type = model_results["problem_type"]
        
        detailed_metrics = {
            "basic_metrics": metrics,
            "model_complexity": {
                "features_used": len(model_results["features"]),
                "model_type": model_results["model_type"]
            }
        }
        
        if problem_type == "classification":
            # Add classification-specific metrics
            detailed_metrics["classification_details"] = {
                "accuracy_difference": metrics["train_accuracy"] - metrics["test_accuracy"],
                "overfitting_indicator": "High" if (metrics["train_accuracy"] - metrics["test_accuracy"]) > 0.1 else "Low"
            }
        else:
            # Add regression-specific metrics
            detailed_metrics["regression_details"] = {
                "r2_difference": metrics["train_r2"] - metrics["test_r2"],
                "rmse_ratio": metrics["test_rmse"] / metrics["train_rmse"] if metrics["train_rmse"] > 0 else 1,
                "overfitting_indicator": "High" if (metrics["train_r2"] - metrics["test_r2"]) > 0.1 else "Low"
            }
        
        return detailed_metrics
    
    def _generate_final_report(self, model_results: dict, evaluation_results: dict) -> dict:
        """Generate comprehensive final report using AI"""
        try:
            performance_summary = evaluation_results["performance_summary"]
            detailed_metrics = evaluation_results["detailed_metrics"]
            
            prompt = f"""
            Generate a comprehensive machine learning model evaluation report:
            
            Model Details:
            - Type: {performance_summary['model_type']}
            - Problem: {performance_summary['problem_type']}
            - Features: {performance_summary['features_count']}
            - Target: {performance_summary['target_variable']}
            
            Performance:
            - Performance Tier: {performance_summary['performance_tier']}
            - Overfitting: {detailed_metrics.get('classification_details', detailed_metrics.get('regression_details', {})).get('overfitting_indicator', 'Unknown')}
            
            Create a professional summary with:
            1. Executive Summary
            2. Model Performance Analysis
            3. Key Findings
            4. Recommendations
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            return {
                "ai_generated_report": response.choices[0].message.content,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_summary": performance_summary,
                "evaluation_status": "Completed Successfully"
            }
            
        except Exception as e:
            return {
                "ai_generated_report": f"Error generating report: {str(e)}",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_summary": evaluation_results["performance_summary"],
                "evaluation_status": "Completed with Errors"
            }
    
    def _generate_recommendations(self, model_results: dict, evaluation_results: dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        performance_tier = evaluation_results["performance_summary"]["performance_tier"]
        detailed_metrics = evaluation_results["detailed_metrics"]
        
        # General recommendations based on performance
        if performance_tier == "Poor":
            recommendations.extend([
                "🔄 Consider collecting more training data",
                "🛠️ Try different algorithms (XGBoost, Neural Networks)",
                "🔧 Perform extensive feature engineering",
                "📊 Review data quality and preprocessing steps"
            ])
        elif performance_tier == "Fair":
            recommendations.extend([
                "⚙️ Tune hyperparameters using GridSearch or RandomSearch",
                "🎯 Try ensemble methods to improve performance",
                "📈 Consider feature selection techniques",
                "🔍 Analyze prediction errors for insights"
            ])
        elif performance_tier == "Good":
            recommendations.extend([
                "✨ Fine-tune hyperparameters for optimal performance",
                "🚀 Consider model deployment preparation",
                "📝 Document model for production use",
                "🔄 Set up model monitoring pipeline"
            ])
        else:  # Excellent
            recommendations.extend([
                "🎉 Model is ready for production deployment",
                "📊 Implement model monitoring and drift detection",
                "🔄 Set up automated retraining pipeline",
                "📋 Create comprehensive model documentation"
            ])
        
        # Overfitting recommendations
        overfitting = detailed_metrics.get('classification_details', detailed_metrics.get('regression_details', {})).get('overfitting_indicator', 'Unknown')
        if overfitting == "High":
            recommendations.extend([
                "⚠️ Address overfitting with regularization techniques",
                "📉 Reduce model complexity or feature count",
                "🔄 Increase training data if possible",
                "✂️ Apply cross-validation more rigorously"
            ])
        
        return recommendations[:8]  # Limit to top 8 recommendations