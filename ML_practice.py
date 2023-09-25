
import csv
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision as tv 
from torch.utils.data import DataLoader
train_data = tv.datasets.MNIST('MNIST/', train = True, transform=None, download=True)
print(len(train_data))
for i in range(5,7):
    x = train_data.data[i]
    plt.imshow(x)
    plt.show()
# csvpath = './cleaned_all_phones.csv'
# data = pd.read_csv(csvpath).sample(frac = 1, random_state = 2).reset_index(drop=True)
# train_data = data[:700]
# valid_data = data[700:]

# # train_data.plot.scatter(y = 'price(USD)', x = 'battery')
# # plt.show()

# feature = ['battery']
# X = train_data[feature]
# Y = train_data['price(USD)']

# polyfeatures = PolynomialFeatures(degree=3)
# X_poly = polyfeatures.fit_transform(X)

# lrmodel = LinearRegression(fit_intercept=True)
# lrmodel.fit(X_poly, Y)
# print(lrmodel.coef_)
# X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# X_curve_poly = polyfeatures.transform(X_curve)

# Y_curve_pred = lrmodel.predict(X_curve_poly)

# plt.scatter(X, Y, label='訓練數據')
# plt.plot(X_curve, Y_curve_pred, color='red', label='多項式回歸擬合')
# plt.legend()
# plt.xlabel('battery')
# plt.ylabel('price(USD)')
# plt.show()

# # print("b_0 = {:.4f}".format(lrmodel.intercept_))
# # print("b_1 = {:.4f}".format(lrmodel.coef_[0]))
