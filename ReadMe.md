# House Price Prediciton

## Use scikit-learn to predict house prices based on features like size, number of rooms, and location.

## Skills: Data preprocessing, linear regression, model evaluation.

### File structure

│── src/ # Source code for model training
│ ├── preprocess.py # Data cleaning & preprocessing
│ ├── model.py # Linear regression model
│ ├── train.py # Training script
│ ├── predict.py # Prediction script
│── data/ # Dataset
│ ├── housingPrice.csv # Testing data

### Key Functions I will need to use:

| Library                      | Function                         | Purpose                                    |
| ---------------------------- | -------------------------------- | ------------------------------------------ |
| pandas                       | read_csv()                       | Load dataset                               |
| pandas                       | dropna()                         | Handle missing values                      |
| pandas                       | get_dummies()                    | Convert categorical variables to numerical |
| numpy                        | array()                          | Handle numerical operations                |
| matplotlib.pyplot            | plot() & scatter()               | Visualization                              |
| seaborn                      | heatmap()                        | Correlation analysis                       |
| scikit-learn.preprocessing   | StandardScaler()                 | Feature scaling                            |
| scikit-learn.model_selection | train_test_split()               | Split dataset                              |
| scikit-learn.linear_model    | LinearRegression()               | Create model                               |
| scikit-learn.metrics         | mean_squared_error(), r2_score() | Evaluate model                             |
