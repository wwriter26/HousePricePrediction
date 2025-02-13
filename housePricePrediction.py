# size, number of rooms, and location.
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocess import loadData, preProcessData
from src.model import splitData, trainModel, evaluateModel
from src.predict import predict

df = loadData("data/Fake_House_Price_Data.csv")
print("\nOriginal columns:")
print(df.columns.tolist())

df = preProcessData(df)

print("\n after columns:")
print(df.columns.tolist())

xTrain, xTest, yTrain, yTest = splitData(df,'price')  # lowercase 'price'

model = trainModel(xTrain, yTrain)

yPred = predict(model, xTest) # This is using all of the fields in xTrain to predict the price 

mse, r2 = evaluateModel(model, xTest, yTest)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


# allow the user to enter in a info for a house and then it will predict the price
user_values = {
  'house_size': float(input("Enter house size (sq ft): ")),
  'num_bedrooms': int(input("Number of bedrooms: ")),
  'num_bathrooms': float(input("Number of bathrooms: ")),
  'garage_spaces': int(input("Number of garage spaces: ")),
  'year_built': int(input("Year built: ")),
  'location_Countryside': 0,
  'location_Downtown': 0,
  'location_Suburb': 0,
}
location = input("Location (Countryside/Downtown/Suburb): ").strip()
if location in ['Countryside', 'Downtown', 'Suburb']:
  user_values[f'location_{location}'] = 1
else:
  print("Invalid location! Please choose Countryside, Downtown, or Suburb")
  exit()

user_df = pd.DataFrame([user_values])
user_pred = predict(model, user_df)

print(f"Predicted Price: ${user_pred[0]:,.2f}")

# Graph of the predicted price vs the actual price based on the house sizes
plt.scatter(xTest['house_size'], yTest, color='blue', label="Actual Prices")
plt.scatter(xTest['house_size'], yPred, color='red', label="Predicted Prices")
plt.xlabel("Size")
plt.ylabel("Price")
plt.legend()
plt.show()
