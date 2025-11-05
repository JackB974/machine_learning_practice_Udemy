#Importing the libraries

#as ==> shortcut // numpy work w/ arrays
import numpy as np
# pandas ==< work w/ dataset / create matrix
import pandas as pd
#matplotlib : plot charts/graph
import matplotlib.pyplot as plt


# Importing the dataset
 
dataset = pd.read_csv("Data.csv")
#print(dataset)
# ==> iloc[rows: colum] ==> so here: take all rows and we'll need onyl three first columns
# x = dataset.iloc[:, [0, 1, 2]] or simply:
x = dataset.iloc[:, :-1].values #takes every values except alst colum
#take alst (dependable variabe = thing we are trying to predict)
y = dataset.iloc[:, -1].values
print(y)


