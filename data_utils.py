import pandas as pd
import json
import os
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def save_state(state: dict, filename: str):
    """Save pipeline state to JSON file"""
    try:
        # Convert pandas objects to serializable format
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, pd.DataFrame):
                serializable_state[key] = {
                    "type": "DataFrame",
                    "shape": value.shape,
                    "columns": value.columns.tolist(),
                    "sample": value.head().to_dict()
                }
            elif hasattr(value, 'to_dict'):
                serializable_state[key] = value.to_dict()
            else:
                serializable_state[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_state, f, indent=2, default=str)
        
        print(f"✅ State saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving state: {str(e)}")

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    data = {
        'feature_1': np.random.normal(50, 15, n_samples),
        'feature_2': np.random.normal(100, 20, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.exponential(2, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'feature_1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_2'] = np.nan
    
    return df