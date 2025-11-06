#Importing the libraries

#as ==> shortcut // numpy work w/ arrays
import numpy as np
# pandas ==< work w/ dataset / create matrix
import pandas as pd
#matplotlib : plot charts/graph
import matplotlib.pyplot as plt
#scikitlearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
 
dataset = pd.read_csv("Data.csv")
#print(dataset)
# ==> iloc[rows: colum] ==> so here: take all rows and we'll need onyl three first columns
# x = dataset.iloc[:, [0, 1, 2]] or simply:
x = dataset.iloc[:, :-1].values #takes every values except alst colum
#take alst (dependable variabe = thing we are trying to predict)
y = dataset.iloc[:, -1].values
#print(y)
#Handle missing data
#replacing missing data by average of all salary for example
#create object of simpleimputer class
#np.nan ==> numpy lib for missing values
# ==> replace all missing value by the mean (strategy = replace missing value by...)
#SimpleImputer(missingvalues, replace by)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#now precise on which colum/row to apply replacement ==> fit (var on wich to applu // []==> the iloc used on x)
#Tip: include all numerical columns values
imputer.fit(x[:, 1:3])
#do the deed! w/ transform
#trasnform ==> return colulns with replacement
#so here says replace x by the transformed imputer of X
x[:, 1:3] = imputer.transform(x[:, 1:3])

#do'nt print umputer print the X, the columns/tableau we did the transform and such on
print(x)

#Encoding Categorical Data
#Country = str / how to transform them for machine elarning to udnerstand it?
# ==> transform them into columns
# creating binary vector for each one
#Called one-hot encoding
 ## ==> transformers :  [("encoder" = label, OneHotEncoder() = the actual encoder, [0] ==> the actual column to transform), passtrough ==> leave other columns unchanged]
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder='passthrough')
#fit method ==> apply Columntransformer
#have to trasnform intoa rray because model amchine ealrning need an array ot ealrn
x = np.array(ct.fit_transform(x))
