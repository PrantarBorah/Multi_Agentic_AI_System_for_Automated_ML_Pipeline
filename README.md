## README.md
```markdown
# 🤖 AI Agent Data Pipeline

An intelligent, multi-agent system that automates the entire machine learning pipeline from data cleaning to model evaluation using CrewAI orchestration and OpenAI GPT-4o.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for each pipeline stage
- **Automated Data Cleaning**: Handle missing values, outliers, and data types
- **Intelligent EDA**: AI-powered exploratory data analysis with visualizations
- **Smart Model Selection**: Automatic algorithm selection based on problem type
- **Comprehensive Evaluation**: Detailed performance metrics and recommendations
- **AI-Powered Insights**: GPT-4o generates insights and recommendations throughout

## 🏗️ Architecture

The pipeline consists of four specialized agents:

1. **CleanerAgent**: Data preprocessing and cleaning
2. **EDAAgent**: Exploratory data analysis and visualization
3. **ModelTrainerAgent**: Model selection, training, and optimization
4. **EvaluatorAgent**: Performance evaluation and reporting

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone or download the codebase
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

1. **Basic Usage** (with sample data):
   ```bash
   python app.py
   ```

2. **With your own data**:
   ```python
   from app import DataPipelineOrchestrator
   
   orchestrator = DataPipelineOrchestrator("path/to/your/data.csv")
   orchestrator.run_pipeline()
   ```

## 📊 Supported Data Formats

- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)

## 🤖 Agent Details

### CleanerAgent
- Handles missing values (median/mode imputation)
- Removes outliers using IQR method
- Fixes data types automatically
- Provides AI-generated cleaning summary

### EDAAgent
- Generates comprehensive summary statistics
- Creates correlation matrices and visualizations
- Detects target variable and problem type
- Provides AI-powered insights about data quality

### ModelTrainerAgent
- Automatically selects appropriate algorithms
- Supports both classification and regression
- Performs model comparison using cross-validation
- Includes feature scaling and preprocessing

### EvaluatorAgent
- Comprehensive performance evaluation
- Creates visualization plots
- Generates detailed metrics
- Provides actionable recommendations

## 📈 Output Files

The pipeline generates several output files:

- `pipeline_results.json`: Complete pipeline state and results
- `correlation_heatmap.png`: Feature correlation visualization
- `distributions.png`: Feature distribution plots
- `classification_evaluation.png` or `regression_evaluation.png`: Model performance plots
- `trained_model_*.joblib`: Saved trained model

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for AI-powered insights

### Customization

You can customize the pipeline by:

1. **Modifying agent behavior**: Edit individual agent classes
2. **Adding new agents**: Create new agents following the existing pattern
3. **Changing orchestration**: Modify the CrewAI setup in `app.py`
4. **Adding algorithms**: Extend the model selection in `ModelTrainerAgent`

## 📝 Example Output

```
🚀 Starting AI Agent Data Pipeline...

🧹 Step 1: Data Cleaning...
📋 Original data shape: (1000, 5)
✅ Cleaned data shape: (1000, 5)

📊 Step 2: Exploratory Data Analysis...
📊 Generating summary statistics...
🔍 Analyzing data types...
🔗 Computing correlations...
📈 Creating visualizations...
🎯 Detecting target variable...
🧠 Generating AI insights...

🤖 Step 3: Model Training...
🎯 Training classification model...
📊 RandomForest CV Score: 0.8234
📊 LogisticRegression CV Score: 0.7891
✅ Model training completed: RandomForestClassifier

📈 Step 4: Model Evaluation...
📊 Generating performance summary...
📈 Creating evaluation visualizations...
🔍 Computing detailed metrics...
📋 Generating final report...
💡 Creating recommendations...

✅ Pipeline completed successfully!
💾 Results saved to pipeline_results.json
```

## 🎯 Use Cases

- **Data Science Automation**: Automate repetitive ML pipeline tasks
- **Rapid Prototyping**: Quickly test ML feasibility on new datasets
- **Educational Tool**: Learn ML pipeline best practices
- **Baseline Model Creation**: Generate baseline models for comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenAI API key is correctly set in `.env`
2. **Package Installation**: Run `pip install -r requirements.txt` in activated environment
3. **Data Loading Error**: Check file path and format support
4. **Memory Issues**: For large datasets, consider data sampling

### Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed error information

## 🔮 Future Enhancements

- Support for more data formats (Parquet, SQL databases)
- Advanced feature engineering agents
- Hyperparameter optimization agent
- Model deployment agent
- Real-time monitoring capabilities
- Integration with MLflow for experiment tracking
```

