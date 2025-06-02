
import os
import json
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from agents.cleaner_agent import CleanerAgent
from agents.eda_agent import EDAAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.evaluator_agent import EvaluatorAgent
from utils.data_utils import load_data, save_state

load_dotenv()

class DataPipelineOrchestrator:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.pipeline_state = {
            "original_data": None,
            "cleaned_data": None,
            "eda_results": None,
            "model_results": None,
            "evaluation_results": None
        }
        
        # Initialize agents
        self.cleaner_agent = CleanerAgent()
        self.eda_agent = EDAAgent()
        self.model_trainer_agent = ModelTrainerAgent()
        self.evaluator_agent = EvaluatorAgent()
    
    def run_pipeline(self):
        """Execute the complete data pipeline using CrewAI orchestration"""
        try:
            # Load initial data
            print("🚀 Starting AI Agent Data Pipeline...")
            self.pipeline_state["original_data"] = load_data(self.data_path)
            
            # Define CrewAI agents
            cleaner = Agent(
                role='Data Cleaner',
                goal='Clean and preprocess the dataset',
                backstory='Expert in data cleaning and preprocessing',
                verbose=True
            )
            
            eda_analyst = Agent(
                role='EDA Analyst',
                goal='Perform exploratory data analysis',
                backstory='Expert in statistical analysis and visualization',
                verbose=True
            )
            
            model_trainer = Agent(
                role='Model Trainer',
                goal='Train and optimize machine learning models',
                backstory='Expert in machine learning model development',
                verbose=True
            )
            
            evaluator = Agent(
                role='Model Evaluator',
                goal='Evaluate model performance and generate insights',
                backstory='Expert in model evaluation and performance metrics',
                verbose=True
            )
            
            # Define tasks
            cleaning_task = Task(
                description=f"Clean the dataset: {self.data_path}",
                agent=cleaner
            )
            
            eda_task = Task(
                description="Perform comprehensive EDA on cleaned data",
                agent=eda_analyst
            )
            
            training_task = Task(
                description="Train appropriate ML model based on data characteristics",
                agent=model_trainer
            )
            
            evaluation_task = Task(
                description="Evaluate model performance and generate final report",
                agent=evaluator
            )
            
            # Create crew and execute
            crew = Crew(
                agents=[cleaner, eda_analyst, model_trainer, evaluator],
                tasks=[cleaning_task, eda_task, training_task, evaluation_task],
                verbose=2,
                process=Process.sequential
            )
            
            # Execute pipeline steps
            self._execute_cleaning()
            self._execute_eda()
            self._execute_training()
            self._execute_evaluation()
            
            print("✅ Pipeline completed successfully!")
            self._save_final_results()
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _execute_cleaning(self):
        """Execute data cleaning step"""
        print("\n🧹 Step 1: Data Cleaning...")
        self.pipeline_state["cleaned_data"] = self.cleaner_agent.clean_data(
            self.pipeline_state["original_data"]
        )
        
    def _execute_eda(self):
        """Execute EDA step"""
        print("\n📊 Step 2: Exploratory Data Analysis...")
        self.pipeline_state["eda_results"] = self.eda_agent.perform_eda(
            self.pipeline_state["cleaned_data"]
        )
        
    def _execute_training(self):
        """Execute model training step"""
        print("\n🤖 Step 3: Model Training...")
        self.pipeline_state["model_results"] = self.model_trainer_agent.train_model(
            self.pipeline_state["cleaned_data"],
            self.pipeline_state["eda_results"]
        )
        
    def _execute_evaluation(self):
        """Execute model evaluation step"""
        print("\n📈 Step 4: Model Evaluation...")
        self.pipeline_state["evaluation_results"] = self.evaluator_agent.evaluate_model(
            self.pipeline_state["model_results"],
            self.pipeline_state["cleaned_data"]
        )
    
    def _save_final_results(self):
        """Save pipeline results"""
        save_state(self.pipeline_state, "pipeline_results.json")
        print(f"💾 Results saved to pipeline_results.json")

if __name__ == "__main__":
    # Example usage
    data_path = "sample_data/sample_dataset.csv"
    orchestrator = DataPipelineOrchestrator(data_path)
    orchestrator.run_pipeline()
