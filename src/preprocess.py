# size, number of rooms, and location.
import pandas as pd


def loadData(filepath):
    df = pd.read_csv(filepath)
    print(df.head())
    return df

def preProcessData(df):
   # Handle missing values
    df = df.dropna()
    
    df = pd.get_dummies(df) # catagorical to numerical
    return df



