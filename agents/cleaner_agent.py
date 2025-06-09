import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import openai
import os

class CleanerAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def clean_data(self, data: pd.DataFrame) -> dict:
        """Clean and preprocess the dataset"""
        cleaned_state = {
            "data": None,
            "cleaning_summary": {},
            "transformations": []
        }
        
        try:
            print(f"📋 Original data shape: {data.shape}")
            
            # Handle missing values
            data_cleaned = self._handle_missing_values(data)
            cleaned_state["transformations"].append("missing_values_handled")
            
            # Handle outliers
            data_cleaned = self._handle_outliers(data_cleaned)
            cleaned_state["transformations"].append("outliers_handled")
            
            # Fix data types
            data_cleaned = self._fix_data_types(data_cleaned)
            cleaned_state["transformations"].append("data_types_fixed")
            
            # Generate cleaning summary using LLM
            cleaning_summary = self._generate_cleaning_summary(data, data_cleaned)
            
            cleaned_state["data"] = data_cleaned
            cleaned_state["cleaning_summary"] = cleaning_summary
            
            print(f"✅ Cleaned data shape: {data_cleaned.shape}")
            return cleaned_state
            
        except Exception as e:
            print(f"❌ Data cleaning failed: {str(e)}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        data_copy = data.copy()
        
        for column in data_copy.columns:
            if data_copy[column].isnull().sum() > 0:
                if data_copy[column].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    data_copy[column].fillna(data_copy[column].mode()[0], inplace=True)
                else:
                    # Fill numerical with median
                    data_copy[column].fillna(data_copy[column].median(), inplace=True)
        
        return data_copy
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        data_copy = data.copy()
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            data_copy[column] = np.clip(data_copy[column], lower_bound, upper_bound)
        
        return data_copy
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data types automatically"""
        data_copy = data.copy()
        
        for column in data_copy.columns:
            # Try to convert to numeric if possible
            if data_copy[column].dtype == 'object':
                try:
                    data_copy[column] = pd.to_numeric(data_copy[column])
                except (ValueError, TypeError):
                    # Keep as categorical if conversion fails
                    pass
        
        return data_copy
    
    def _generate_cleaning_summary(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict:
        """Generate cleaning summary using LLM"""
        try:
            prompt = f"""
            Analyze the data cleaning process:
            - Original shape: {original_data.shape}
            - Cleaned shape: {cleaned_data.shape}
            - Missing values before: {original_data.isnull().sum().sum()}
            - Missing values after: {cleaned_data.isnull().sum().sum()}
            
            Provide a brief summary of the cleaning process and its impact.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            return {
                "ai_summary": response.choices[0].message.content,
                "missing_values_removed": original_data.isnull().sum().sum(),
                "outliers_handled": True,
                "data_types_optimized": True
            }
            
        except Exception as e:
            return {
                "ai_summary": f"Error generating summary: {str(e)}",
                "missing_values_removed": original_data.isnull().sum().sum(),
                "outliers_handled": True,
                "data_types_optimized": True
            }