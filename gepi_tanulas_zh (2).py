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

df

#2. feladat

iris_by_target = df.groupby(by='Wristcircumference');  # grouping by target
# Exporting a target group from the grouped dataframe


# Basic descriptive stats
mean_by_target = iris_by_target.mean();  # mean
std_by_target = iris_by_target.std();  # standard deviation
corr_by_target = iris_by_target.corr();  #  correlations
desc_stat_by_target = iris_by_target.describe();  # desc stat with quantiles

# Printing the results
print(mean_by_target);

#3. feladat matrix plot


plt.figure(1);
pd.plotting.scatter_matrix(df[['Forearmcircumference','Density','Age','Weight']]);
plt.show();

#4. feladat : Filter I,O

X_train, X_test, y_train, y_test = train_test_split(df, diabetes.target, 
                                                    test_size=0.2, random_state=2022)

#5.feladat

class_tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4);
class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train);
score_test_tree = class_tree.score(X_test, y_test);


logreg_classifier = LogisticRegression(solver = 'newton-cg');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);
ypred_logreg = logreg_classifier.predict(X_test);
yprobab_logreg = logreg_classifier.predict_proba(X_test);


neural_classifier = MLPClassifier(hidden_layer_sizes=(3),
                                  activation='logistic',
                                  max_iter=1000);
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);
score_test_neural = neural_classifier.score(X_test,y_test);
ypred_neural = neural_classifier.predict(X_test);
yprobab_neural = neural_classifier.predict_proba(X_test);

print(f'Test score of tree in %: {score_test_tree*100}');
print(f'Test score of logreg in %: {score_test_logreg*100}'); 
print(f'Test score of neural in %: {score_test_neural*100}');

#6.feladat:

    

cm_logreg_test = metrics.confusion_matrix(y_test, ypred_logreg);

fpr_logreg, tpr_logreg, _ = metrics.roc_curve(y_test, yprobab_logreg[:,0], pos_label=0);
roc_auc_logreg = metrics.auc(fpr_logreg, tpr_logreg);

plt.figure(3);
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=2, label='Logreg (area = %0.2f)' % roc_auc_logreg);
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

#7.feladat: Nem felügyelt

without_letter = df.get(['x-box','y-box','width','high','onpix',
                   'x-bar', 'y-bar', 'x2bar','y2bar','xybar', 'x2ybr','xy2br',
                   'x-ege','xegvy','y-ege','yegvx'])

kmeans10 = KMeans(n_clusters=10, random_state=2021);
kmeans10.fit(without_letter);
labels10 = kmeans10.labels_;
centers10 = kmeans10.cluster_centers_;
DB10 = davies_bouldin_score(without_letter,labels10);

kmeans26 = KMeans(n_clusters=26, random_state=2021);
kmeans26.fit(without_letter);
labels26 = kmeans26.labels_;
centers26 = kmeans26.cluster_centers_;
DB26 = davies_bouldin_score(without_letter,labels26);

kmeans30 = KMeans(n_clusters=30, random_state=2021);
kmeans30.fit(without_letter);
labels30 = kmeans30.labels_;
centers30 = kmeans30.cluster_centers_;
DB30 = davies_bouldin_score(without_letter,labels30);

print(f'10 -es Klaszterrel a DB : {DB10}')
print(f'26 -os Klaszterrel a DB : {DB26}')
print(f'30 -as Klaszterrel a DB : {DB30}')

pca = PCA(n_components=2);
pca.fit(without_letter);
data_pc = pca.transform(without_letter);
centers_pc = pca.transform(centers26);

Max_K = 31;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(without_letter);
    bc_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(without_letter,bc_labels);
    
# Visualization of SSE values    
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

fig = plt.figure(4);
plt.title('Clustering of the Letter data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(data_pc[:,0],data_pc[:,1],s=50,c=labels26);
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');
plt.show();

