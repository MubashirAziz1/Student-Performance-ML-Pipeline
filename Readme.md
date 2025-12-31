# 🎓 Student Performance Predictor - End-to-End Machine Learning Project

A comprehensive machine learning project that predicts student math performance based on demographic factors, parental education level, and existing academic scores. This project includes a complete ML pipeline from data ingestion to model deployment with a user-friendly web interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Project Workflow](#project-workflow)
- [Author](#author)

## 🎯 Overview

This project implements an end-to-end machine learning solution for predicting student math scores. It uses various machine learning algorithms including Random Forest, XGBoost, CatBoost, and others, with hyperparameter tuning to find the best-performing model. The best model is tracked and managed using MLflow, and predictions can be made through a Flask web application.

## ✨ Features

- **Comprehensive ML Pipeline**: Complete workflow from data ingestion to model deployment
- **Multiple Model Comparison**: Tests 7 different machine learning algorithms
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **MLflow Integration**: Model versioning, tracking, and management
- **Web Interface**: User-friendly Flask web application for predictions
- **Data Preprocessing**: Automated data transformation pipeline
- **Model Persistence**: Save and load trained models and preprocessors
- **Docker Support**: Containerized deployment option

## 📁 Project Structure

```
code/
├── artifacts/                 # Saved models, preprocessors, and data
│   ├── model.pkl             # Trained model
│   ├── preprocessor.pkl      # Data preprocessor
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Testing dataset
│   └── data.csv              # Raw dataset
├── src/                       # Source code
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Data preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/             # Prediction pipeline
│   │   ├── predict_pipeline.py   # Prediction logic
│   │   └── train_pipeline.py     # Training pipeline
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Logging configuration
│   └── utils.py              # Utility functions
├── templates/                # HTML templates
│   ├── index.html            # Home page
│   └── home.html             # Prediction form
├── notebook/                 # Jupyter notebooks for EDA and training
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
├── logs/                     # Application logs
├── mlruns/                   # MLflow experiment tracking
├── app.py                    # Flask web application
├── main.py                   # Training script
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── Dockerfile               # Docker configuration
└── Readme.md                # This file
```

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn** (1.3.2): Model training and preprocessing
- **CatBoost** (1.2.2): Gradient boosting algorithm
- **XGBoost** (1.7.6): Extreme gradient boosting
- **MLflow** (2.9.2): Model tracking and versioning

### Data Processing
- **pandas** (2.1.4): Data manipulation
- **numpy** (1.26.4): Numerical computations

### Visualization
- **matplotlib** (3.8.2): Plotting
- **seaborn** (0.13.1): Statistical visualization

### Web Framework
- **Flask** (2.3.3): Web application framework

### Utilities
- **dill** (0.3.7): Object serialization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "End_to_End ML Project/code"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install the Package
```bash
pip install -e .
```

## 🚀 Usage

### Training the Model

To train the model from scratch, run:

```bash
python main.py
```

This will:
1. Load the dataset from `notebook/data/stud.csv`
2. Split the data into training and testing sets (80/20)
3. Preprocess the data (handle missing values, encode categorical variables, scale features)
4. Train multiple models with hyperparameter tuning
5. Select the best model based on R² score
6. Save the best model and preprocessor to `artifacts/`
7. Log all experiments to MLflow

**Note**: Ensure that the dataset file exists at `notebook/data/stud.csv` before running the training script.

### Running the Web Application

To start the Flask web application:

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000` (or `http://localhost:5000`).

#### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Click on "Predict Performance" button
3. Fill in the form with student information:
   - **Gender**: Male or Female
   - **Race/Ethnicity**: Group A, B, C, D, or E
   - **Parental Level of Education**: Various education levels
   - **Lunch Type**: Free/Reduced or Standard
   - **Test Preparation Course**: None or Completed
   - **Reading Score**: Score between 0-100
   - **Writing Score**: Score between 0-100
4. Click "Predict Your Math Score"
5. View the predicted math score

## 🤖 Model Information

### Models Evaluated
The project evaluates the following machine learning algorithms:

1. **Random Forest Regressor**
2. **Decision Tree Regressor**
3. **Gradient Boosting Regressor**
4. **Linear Regression**
5. **XGBoost Regressor**
6. **CatBoost Regressor**
7. **AdaBoost Regressor**

### Model Selection
- Models are evaluated using **R² Score** (Coefficient of Determination)
- **GridSearchCV** with 3-fold cross-validation is used for hyperparameter tuning
- The best model is selected based on test set performance
- Minimum acceptable R² score threshold: 0.6

### Features Used
- **Categorical Features**:
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch Type
  - Test Preparation Course

- **Numerical Features**:
  - Reading Score
  - Writing Score

- **Target Variable**:
  - Math Score

### Preprocessing Pipeline
1. **Categorical Variables**:
   - Missing value imputation (most frequent)
   - One-hot encoding
   - Standard scaling

2. **Numerical Variables**:
   - Missing value imputation (median)
   - Standard scaling

## 🌐 API Endpoints

### Home Page
- **URL**: `/`
- **Method**: GET
- **Description**: Displays the welcome page with project information

### Prediction Page
- **URL**: `/predictdata`
- **Methods**: GET, POST
- **GET**: Displays the prediction form
- **POST**: Submits form data and returns prediction
- **Form Fields**:
  - `gender` (string)
  - `ethnicity` (string)
  - `parental_level_of_education` (string)
  - `lunch` (string)
  - `test_preparation_course` (string)
  - `reading_score` (float)
  - `writing_score` (float)

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t student-performance-predictor .
```

### Run Docker Container
```bash
docker run -p 5000:5000 student-performance-predictor
```

The application will be accessible at `http://localhost:5000`

## 📊 Project Workflow

```
1. Data Ingestion
   └──> Load data from CSV
   └──> Split into train/test sets
   └──> Save to artifacts/

2. Data Transformation
   └──> Create preprocessing pipeline
   └──> Handle missing values
   └──> Encode categorical variables
   └──> Scale numerical features
   └──> Save preprocessor

3. Model Training
   └──> Train multiple models
   └──> Hyperparameter tuning (GridSearchCV)
   └──> Evaluate models (R² score)
   └──> Select best model
   └──> Log to MLflow
   └──> Save best model

4. Prediction
   └──> Load preprocessor
   └──> Load model (from MLflow or artifacts)
   └──> Transform input data
   └──> Make prediction
   └──> Return result
```

## 📝 Logging

The project includes comprehensive logging:
- Logs are saved in the `logs/` directory
- Each run creates a timestamped log file
- Logs include:
  - Data ingestion progress
  - Data transformation steps
  - Model training progress
  - Error messages and exceptions

## 🔍 MLflow Tracking

The project uses MLflow for experiment tracking:
- **Experiment Name**: "Student Performance Predictor"
- **Tracked Metrics**: 
  - Train R² Score
  - Test R² Score
- **Tracked Parameters**: 
  - Model type
  - Best hyperparameters for each model
- **Model Registry**: Best model is registered in MLflow model registry

To view MLflow UI:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser (if Flask app is not running) or use a different port:
```bash
mlflow ui --port 5001
```

## ⚠️ Important Notes

1. **Data Path**: Ensure the dataset file exists at `notebook/data/stud.csv` before training
2. **MLflow Model**: The prediction pipeline loads models from MLflow registry. Make sure MLflow is properly configured
3. **Artifacts Directory**: The `artifacts/` directory is created automatically during training
4. **Model Files**: Trained models and preprocessors are saved in the `artifacts/` directory

## 🐛 Troubleshooting

### Issue: Module not found
**Solution**: Make sure you've installed the package using `pip install -e .`

### Issue: Dataset not found
**Solution**: Ensure `notebook/data/stud.csv` exists or update the path in `data_ingestion.py`

### Issue: MLflow model not found
**Solution**: 
- Train the model first using `python main.py`
- Or update `predict_pipeline.py` to load from `artifacts/model.pkl` instead

### Issue: Port already in use
**Solution**: Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Mubashir**
- Email: cs.mubashir.a@gmail.com

## 🙏 Acknowledgments

- scikit-learn community for excellent ML tools
- Flask team for the web framework
- MLflow team for model tracking capabilities

---

**Note**: This is an end-to-end machine learning project demonstrating best practices in ML engineering, including proper project structure, logging, exception handling, and model deployment.
