
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Splitting data into training and testing sets
def splitData(df, trainingCol):
  x = df.drop(trainingCol, axis=1)
  y = df[trainingCol]
  return train_test_split(x, y, test_size=0.2, random_state=42)
  

def trainModel(xTrain, yTrain):
  model = LinearRegression()
  model.fit(xTrain, yTrain)
  return model

# This evaluates the model by calculating the mean squared error and r2 score
def evaluateModel(model, xTest, yTest):
  yPred = model.predict(xTest)
  mse = mean_squared_error(yTest, yPred)
  r2 = r2_score(yTest, yPred)
  return mse, r2
