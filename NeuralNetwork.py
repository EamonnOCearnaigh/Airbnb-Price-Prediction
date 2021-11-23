# -*- coding: utf-8 -*-
"""
Introduction to AI Coursework: Neural Network for Airbnb Price Analysis & Prediction

@authors: Éamonn Ó Cearnaigh
"""

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =-=-=-=-=-=-=-=-=-=-=-=
# Testing Neural Network
# =-=-=-=-=-=-=-=-=-=-=-=

# Status: Arranging components, need to extensively clean data for it to work.

def plot_corr(df,size=10):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    plt.figure()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    
    
# Input listings, neigbourhoods, reviews - Airbnb datasets
listings = pd.read_csv('Airbnb/listings.csv')
print (listings.head())
neighbourhoods = pd.read_csv('Airbnb/neighbourhoods.csv')
print (neighbourhoods.head())
reviews = pd.read_csv('Airbnb/reviews.csv')
print (reviews.head())

# Neighbourhoods and reviews have no meaningful correlation within themselves.
plot_corr(listings)
# Listings show potential correlations.


# Set X and y
# Train test split 
X, y = listings.drop('price', axis=1), listings['price'];
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Number of hidden layers needed:
# Nh = Ns/(α∗ (Ni + No))
# Ni = number of input neurons.
# No = number of output neurons.
# Ns = number of samples in training data set.
# α = an arbitrary scaling factor usually 2-10. (???)

# ...



# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Continue...

# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

