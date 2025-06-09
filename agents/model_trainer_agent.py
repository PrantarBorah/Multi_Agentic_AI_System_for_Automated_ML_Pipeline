import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import openai
import os

class ModelTrainerAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def train_model(self, cleaned_state: dict, eda_results: dict) -> dict:
        """Train machine learning model based on problem type"""
        data = cleaned_state["data"]
        problem_type = eda_results["problem_type"]
        target_column = eda_results["target_variable"]
        
        model_results = {
            "model": None,
            "model_type": None,
            "features": [],
            "target": target_column,
            "problem_type": problem_type,
            "preprocessing": {},
            "training_metrics": {},
            "model_path": None
        }
        
        try:
            print(f"🎯 Training {problem_type} model...")
            
            # Prepare features and target
            X, y = self._prepare_features_target(data, target_column)
            model_results["features"] = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "classification" else None
            )
            
            # Preprocess features
            X_train_scaled, X_test_scaled = self._preprocess_features(X_train, X_test)
            model_results["preprocessing"]["scaling"] = "StandardScaler applied"
            
            # Select and train model
            model = self._select_and_train_model(X_train_scaled, y_train, problem_type)
            model_results["model"] = model
            model_results["model_type"] = type(model).__name__
            
            # Evaluate model
            training_metrics = self._evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
            model_results["training_metrics"] = training_metrics
            
            # Save model
            model_path = f"trained_model_{problem_type}.joblib"
            joblib.dump(model, model_path)
            model_results["model_path"] = model_path
            
            # Generate AI insights
            model_results["ai_insights"] = self._generate_model_insights(model_results)
            
            print(f"✅ Model training completed: {model_results['model_type']}")
            return model_results
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            raise
    
    def _prepare_features_target(self, data: pd.DataFrame, target_column: str) -> tuple:
        """Prepare features and target variables"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Simple label encoding for categorical variables
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if y.dtype in ['object', 'category']:
            y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def _preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Preprocess features using scaling"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _select_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, problem_type: str):
        """Select and train appropriate model"""
        if problem_type == "classification":
            # Try multiple models and select best one
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "LinearRegression": LinearRegression()
            }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            # Use cross-validation to select best model
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='accuracy' if problem_type == "classification" else 'r2')
            avg_score = cv_scores.mean()
            
            print(f"📊 {name} CV Score: {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train the best model
        best_model.fit(X_train, y_train)
        return best_model
    
    def _evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, problem_type: str) -> dict:
        """Evaluate model performance"""
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if problem_type == "classification":
            return {
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "classification_report": classification_report(y_test, test_pred, output_dict=True)
            }
        else:
            return {
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_mse": mean_squared_error(y_train, train_pred),
                "test_mse": mean_squared_error(y_test, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred))
            }
    
    def _generate_model_insights(self, model_results: dict) -> dict:
        """Generate AI insights about model performance"""
        try:
            metrics = model_results["training_metrics"]
            problem_type = model_results["problem_type"]
            
            if problem_type == "classification":
                performance_summary = f"Test Accuracy: {metrics['test_accuracy']:.4f}"
            else:
                performance_summary = f"Test R²: {metrics['test_r2']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}"
            
            prompt = f"""
            Analyze this {problem_type} model performance:
            - Model: {model_results['model_type']}
            - Performance: {performance_summary}
            - Features: {len(model_results['features'])}
            
            Provide insights about model performance and suggestions for improvement.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )
            
            return {
                "ai_analysis": response.choices[0].message.content,
                "performance_tier": self._classify_performance(metrics, problem_type),
                "improvement_suggestions": self._generate_improvement_suggestions(metrics, problem_type)
            }
            
        except Exception as e:
            return {
                "ai_analysis": f"Error generating insights: {str(e)}",
                "performance_tier": "Unknown",
                "improvement_suggestions": ["Review model selection", "Try hyperparameter tuning"]
            }
    
    def _classify_performance(self, metrics: dict, problem_type: str) -> str:
        """Classify model performance into tiers"""
        if problem_type == "classification":
            accuracy = metrics["test_accuracy"]
            if accuracy > 0.9:
                return "Excellent"
            elif accuracy > 0.8:
                return "Good"
            elif accuracy > 0.7:
                return "Fair"
            else:
                return "Poor"
        else:
            r2 = metrics["test_r2"]
            if r2 > 0.9:
                return "Excellent"
            elif r2 > 0.7:
                return "Good"
            elif r2 > 0.5:
                return "Fair"
            else:
                return "Poor"
    
    def _generate_improvement_suggestions(self, metrics: dict, problem_type: str) -> list:
        """Generate suggestions for model improvement"""
        suggestions = []
        
        if problem_type == "classification":
            if metrics["test_accuracy"] < 0.8:
                suggestions.extend([
                    "Try feature engineering",
                    "Consider ensemble methods",
                    "Tune hyperparameters",
                    "Collect more training data"
                ])
        else:
            if metrics["test_r2"] < 0.7:
                suggestions.extend([
                    "Try polynomial features",
                    "Consider regularization",
                    "Feature selection",
                    "Try different algorithms"
                ])
        
        return suggestions