import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import plotting, preprocess
import functions as fc
import pickle
import os

def main():
  # Loads the data file, if doesnt exists then generates it
  if os.path.exists('data.csv'):
    df = pd.read_csv("data.csv")
  else:
    processer = preprocess.Preprocesser('japanese.csv', 'english.csv')
    df = processer.get_processed_dataframe()
    df.to_csv("data.csv", index=False)

  # Makes the training and testing set
  features = df.columns[3:]
  X = df[features]
  y = df['language_encoded']
  indicies = np.arange(len(y))
  XTrain, XTest, yTrain, yTest, indTrain, indTest = train_test_split(X, y, indicies, test_size=0.01, random_state=0)
  print('Test size: ', len(yTest))

  # Normalizes values to Gaussian 
  sc = StandardScaler()
  sc.fit(XTrain.to_numpy())
  XTrainStd = sc.transform(XTrain.to_numpy())
  XTestStd = sc.transform(XTest.to_numpy())

  ## Principal component analysis
  pca = PCA(n_components=2)
  XTrainPCA = pca.fit_transform(XTrainStd)
  XTestPCA = pca.transform(XTestStd)
  print('Explained variance: ', pca.explained_variance_ratio_)

  # Gets the features with the largest linear component in the tranformation matrix
  transWeight = np.stack((features, np.abs(pca.components_[0]), np.abs(pca.components_[1]))).T
  transWeight = transWeight[np.argsort(transWeight[:, 1])]
  print("Linear transformation weights: ", transWeight)

  # Plots that top 20 features that make up each principle component
  top_weights = transWeight[-20:,:]
  plt.bar(range(len(top_weights)), top_weights.T[1], color='lightblue', align='center')
  plt.xticks(range(len(top_weights)), top_weights.T[0], rotation=90)
  plt.xlim([-1, len(top_weights)])
  plt.tight_layout()
  plt.show()
  plt.title('First PCA Transformation')
  plt.ylabel('Magnitude of transformation weights')
  plt.show()
  plt.savefig('images/pca1_matrix_weights.pdf',bbox_inches='tight')
  plt.savefig('images/pca1_matrix_weights.png',bbox_inches='tight')
  plt.cla()

  transWeight = transWeight[np.argsort(transWeight[:, 2])]
  top_weights_2 = transWeight[-20:,:]
  plt.bar(range(len(top_weights_2)), top_weights_2.T[2], color='green', align='center')
  plt.xticks(range(len(top_weights_2)), top_weights_2.T[0], rotation=90)
  plt.title('Second PCA Transformation')
  plt.ylabel('Magnitude of transformation weights')
  plt.savefig('images/pca2_matrix_weights.pdf',bbox_inches='tight')
  plt.savefig('images/pca2_matrix_weights.png',bbox_inches='tight')
  plt.clf()

  ## Logistic regression
  plotx, plotInd = fc.r_subarr(XTrainPCA, 5000)
  ploty = yTrain.to_numpy()[plotInd]

  logreg = LogisticRegression(C=1000, random_state=0)
  logreg.fit(XTrainPCA, yTrain)
  print('Training accuracy: ', logreg.score(XTrainPCA, yTrain))
  print('Testing accuracy: ', logreg.score(XTestPCA, yTest))

  ### Pickles files for webapp
  dest = os.path.join('.','pickle_objs')
  if not os.path.exists(dest):
    os.makedirs(dest)
  
  pickle.dump(sc, open(os.path.join(dest,'sc_trained.pkl'),'wb'),protocol=4)
  pickle.dump(pca, open(os.path.join(dest,'pca_trained.pkl'),'wb'),protocol=4)
  pickle.dump(logreg, open(os.path.join(dest,'logreg_trained.pkl'),'wb'),protocol=4)

  #Training data
  plotting.plot_2_feature_dr(plotx, ploty, logreg)
  plt.show()
  plt.title('Logistic Regression - Training set')
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.savefig('images/decision_logreg.png',bbox_inches='tight')
  plt.clf()
  #Test data
  plotting.plot_2_feature_dr(XTestPCA, yTest.to_numpy(), logreg)
  plt.show()
  plt.title('Logistic Regression - Test set')
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  plt.savefig('images/decision_logreg_test.png',bbox_inches='tight')
  plt.clf()

if __name__== "__main__":
  main()