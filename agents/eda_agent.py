import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openai
import os

class EDAAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def perform_eda(self, cleaned_state: dict) -> dict:
        """Perform comprehensive exploratory data analysis"""
        data = cleaned_state["data"]
        
        eda_results = {
            "summary_stats": {},
            "data_types": {},
            "correlations": {},
            "visualizations": [],
            "insights": {},
            "target_variable": None,
            "problem_type": None
        }
        
        try:
            print("📊 Generating summary statistics...")
            eda_results["summary_stats"] = self._generate_summary_stats(data)
            
            print("🔍 Analyzing data types...")
            eda_results["data_types"] = self._analyze_data_types(data)
            
            print("🔗 Computing correlations...")
            eda_results["correlations"] = self._compute_correlations(data)
            
            print("📈 Creating visualizations...")
            eda_results["visualizations"] = self._create_visualizations(data)
            
            print("🎯 Detecting target variable...")
            eda_results["target_variable"], eda_results["problem_type"] = self._detect_target_and_problem_type(data)
            
            print("🧠 Generating AI insights...")
            eda_results["insights"] = self._generate_ai_insights(data, eda_results)
            
            return eda_results
            
        except Exception as e:
            print(f"❌ EDA failed: {str(e)}")
            raise
    
    def _generate_summary_stats(self, data: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        return {
            "shape": data.shape,
            "numeric_summary": numeric_data.describe().to_dict(),
            "categorical_summary": {col: data[col].value_counts().head().to_dict() 
                                  for col in categorical_data.columns},
            "missing_values": data.isnull().sum().to_dict()
        }
    
    def _analyze_data_types(self, data: pd.DataFrame) -> dict:
        """Analyze data types and suggest improvements"""
        return {
            "dtypes": data.dtypes.to_dict(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "total_columns": len(data.columns)
        }
    
    def _compute_correlations(self, data: pd.DataFrame) -> dict:
        """Compute correlation matrix for numeric variables"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(corr_matrix)
            }
        else:
            return {"correlation_matrix": {}, "high_correlations": []}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
        """Find highly correlated variable pairs"""
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corrs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
        return high_corrs
    
    def _create_visualizations(self, data: pd.DataFrame) -> list:
        """Create and save visualizations"""
        visualizations = []
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Correlation heatmap
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("correlation_heatmap.png")
            
            # 2. Distribution plots for numeric variables
            if len(numeric_data.columns) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_data.columns[:4]):
                    if i < 4:
                        sns.histplot(data[col], kde=True, ax=axes[i])
                        axes[i].set_title(f'Distribution of {col}')
                
                plt.tight_layout()
                plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append("distributions.png")
            
            return visualizations
            
        except Exception as e:
            print(f"⚠️ Warning: Visualization creation failed: {str(e)}")
            return []
    
    def _detect_target_and_problem_type(self, data: pd.DataFrame) -> tuple:
        """Detect likely target variable and problem type"""
        # Simple heuristic: last column is often the target
        target_column = data.columns[-1]
        
        # Determine problem type based on target variable
        if data[target_column].dtype in ['object', 'category']:
            problem_type = "classification"
        elif len(data[target_column].unique()) < 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
        
        return target_column, problem_type
    
    def _generate_ai_insights(self, data: pd.DataFrame, eda_results: dict) -> dict:
        """Generate AI-powered insights about the dataset"""
        try:
            prompt = f"""
            Analyze this dataset and provide key insights:
            - Shape: {data.shape}
            - Columns: {list(data.columns)}
            - Problem type: {eda_results['problem_type']}
            - Target variable: {eda_results['target_variable']}
            - High correlations: {len(eda_results['correlations']['high_correlations'])}
            
            Provide 3-5 key insights about this dataset for machine learning.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            
            return {
                "ai_insights": response.choices[0].message.content,
                "data_quality_score": self._calculate_data_quality_score(data),
                "recommendations": self._generate_recommendations(eda_results)
            }
            
        except Exception as e:
            return {
                "ai_insights": f"Error generating insights: {str(e)}",
                "data_quality_score": 0.5,
                "recommendations": ["Review data quality", "Check for missing values"]
            }
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple data quality score"""
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        return round(completeness, 2)
    
    def _generate_recommendations(self, eda_results: dict) -> list:
        """Generate recommendations based on EDA"""
        recommendations = []
        
        if eda_results['problem_type'] == "classification":
            recommendations.append("Consider using classification algorithms (Random Forest, XGBoost)")
        else:
            recommendations.append("Consider using regression algorithms (Linear Regression, Random Forest)")
        
        if len(eda_results['correlations']['high_correlations']) > 0:
            recommendations.append("Remove highly correlated features to avoid multicollinearity")
        
        return recommendations