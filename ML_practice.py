import numpy as np
import csv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
csvpath = './cleaned_all_phones.csv'
data = pd.read_csv(csvpath).sample(frac = 1, random_state = 2).reset_index(drop=True)
train_data = data[:700]
test_data = data[700:]
print(train_data)