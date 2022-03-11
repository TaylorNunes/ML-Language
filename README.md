# Text Classification for Japanese and English
Machine learning project to classify alphanumeric sentences into English and Japanese

### Installation
This project requires the following packages:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](http://matplotlib.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)

These can be installed from the requirements.txt file.

``` bash
pip install requirements.txt
```

### Usage 

A script to run the project is located in **main.py** which can be run directly from the terminal with python 
```bash
python main.py
```
The script contains multiple sections:
- Data cleaning and feature generation from csv files
- Separating the data into a training and a test dataset
- Performing feature extraction using principal component analysis
- Training a logistic regression model on the training set
- Confirming the performance with a test set

### Results 

The two languages are linearly separable with 2 feature logistic regression with an accuracy of 99.9%.
You can check out a more detailed description [here](https://info.tcnunes.com/projects/textclassification).


