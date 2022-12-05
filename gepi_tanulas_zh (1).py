# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:02:11 2021

@author: septe
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
import pandas as pd;  # importing pandas data analysis tool
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn import model_selection as ms;
from sklearn.tree import DecisionTreeClassifier, plot_tree; 
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.neural_network import MLPClassifier;
from sklearn.cluster import KMeans;
from sklearn.metrics import davies_bouldin_score;
from sklearn.decomposition import PCA;
from sklearn import metrics;

#1.feladat
url = 'https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/bodyfat.csv';
raw_data = urlopen(url);
attribute_names = np.loadtxt(raw_data, max_rows=1, delimiter=",",dtype=np.str);  # reading the first row with attribute names
data = np.loadtxt(raw_data, delimiter=",");  # reading numerical data from csv file
del raw_data;
# Removing unnecessary "s and spaces from names
for i in range(len(attribute_names)):
    attribute_names[i] = attribute_names[i].replace('"','');
    attribute_names[i] = attribute_names[i].replace(' ','');

# Defining dataframes with column names from numpy array
df = pd.DataFrame(data=data, columns=attribute_names);  #  reading the data into dataframe

print(df);
