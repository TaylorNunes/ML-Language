from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_2_feature_dr(X, y, model, res=0.01, firstIndex=None, secondIndex=None):
  xmargin = 0.05*(X[:,0].max()-X[:,0].min())
  ymargin = 0.05*(X[:,1].max()-X[:,1].min())
  plotX = np.arange(X[:,0].min() - xmargin, X[:,0].max() + xmargin, res)
  plotY = np.arange(X[:,1].min() - ymargin, X[:,1].max() + ymargin, res)
  plotxx, plotyy = np.meshgrid(plotX, plotY)
  Z = model.predict(np.array([plotxx.ravel(), plotyy.ravel()]).T)
  Z = Z.reshape(plotxx.shape)
  markers = ['s','o','x','^','v']
  colors = ['forestgreen','royalblue', 'mediumpurple', 'lightcoral', 'goldenrod']
  myCmap = ListedColormap(colors[:len(np.unique(y))])
  plt.contourf(plotxx, plotyy, Z, cmap=myCmap, alpha=0.5, antialiased=True)
  plt.xlim(plotxx.min(), plotxx.max())
  plt.ylim(plotyy.min(), plotyy.max())

  ind = 0
  labels = ['English','Japanese (Romaji)']
  for cl in np.unique(y):
    """ if ind == 1: 
      ind+=1  
      continue"""
    plt.scatter(X[np.where(y[:] == cl), 0], X[np.where(y[:] == cl), 1], c=colors[ind], edgecolors="black", alpha=0.8, marker=markers[ind], label=labels[cl])
    ind+=1
  plt.legend(loc='upper left')

def plot_pairplot(df):
  sns.set(style="ticks", color_codes=True)
  sns.pairplot(df, hue='language')