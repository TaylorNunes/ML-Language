import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import product

class Preprocesser():
  def __init__(self, csv_path_japanese, csv_path_english):
    self.csv_path_japanese = csv_path_japanese
    self.csv_path_english = csv_path_english

  def get_processed_dataframe(self, nSentences=100000):
    df_j = pd.read_csv(self.csv_path_japanese, dtype="string")
    df_e = pd.read_csv(self.csv_path_english, dtype="string")
    
    df = self.__reduce_and_combine(df_j, df_e, nSentences)
    df = self.__clean_text(df)
    df = self.__add_columns(df)
    
    df.drop(columns="id", inplace=True)
    df['language_encoded'] = LabelEncoder().fit_transform(df['language'])
    df['language'].replace("jpn","japanese", inplace=True)
    df['language'].replace("eng","english", inplace=True)
    return df

  def __reduce_and_combine(self, df_j, df_e, nSentences):
    maxAllowable = min(len(df_j),len(df_e))
    if nSentences > maxAllowable: 
      nSentences = maxAllowable
      print('Only {} sentences were used'.format(nSentences))
    df_j = df_j.iloc[:nSentences,:]
    df_e = df_e.iloc[:nSentences,:]
    
    df_j.rename(columns={"Column1":"id","Column2":"language","Column3":"text"}, inplace=True)
    df_e.rename(columns={"Column1":"id","Column2":"language","Column3":"text"}, inplace=True)
    return pd.concat([df_j, df_e])

  def __clean_text(self, df):
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace('[^\w\s]','', regex=True)
    df["text"] = df["text"].str.replace('[\d+]','', regex=True)
    df["text"] = df["text"].str.strip()
    df = df[df['text'] != ""]
    return df

  def __add_columns(self, df):
    letter_freq_list = 'abcdefghijklmnopqrstuvwxyz '
    consecutive_letters = [''.join(p) for p in product(letter_freq_list, repeat=2)]

    feature_df_list = []
    for letter in letter_freq_list:
      df['{}_density'.format(letter)] = df['text'].apply(lambda x : self.char_density(x,letter))  
    for comb in consecutive_letters:
      df['{}_density'.format(comb)] = df['text'].apply(lambda x : self.consec_density(x,comb)) 
    df = df.copy()
    return df
  
  def char_density(self, text, letter):
    return len([a for a in text if a.casefold() == letter])/sum([len(b) for b in text.split()])

  def consec_density(self, text, comb):
    return text.count(comb)/len(text)

