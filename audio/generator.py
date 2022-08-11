import tensorflow_io as tfio
import tensorflow as tf
import pandas as pd
import numpy as np
from preprocess import ShortAudioPreprocesser

class MyGenerator(tf.keras.utils.Sequence):

  def __init__(self, df,
                batch_size,
                clip_length_seconds,
                shuffle=True,
                rate=22050,
                spec_nfft=512,
                spec_window=2048,
                spec_stride=256,
                spec_mels=80):
    
    self.df = df.copy()
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.n = len(df['file_path'])//batch_size
    self.rate=rate
    self.audio_processer = ShortAudioPreprocesser(clip_length_seconds, self.rate, spec_nfft, spec_window, spec_stride, spec_mels)
    shape_spectrogram = self.audio_processer.get_spectrogram(np.ones(self.rate*clip_length_seconds))
    self.spectrogram_shape = np.expand_dims(shape_spectrogram, axis=shape_spectrogram.ndim).shape

  def __get_batch(self, df):
    spectrogram_list = [] 
    label_list = []
    for index, row in df.iterrows():
      audio = self.audio_processer.get_standard_audio(row['file_path'])
      spectrogram = self.audio_processer.get_spectrogram(tf.convert_to_tensor(audio))
      spectrogram_list.append(spectrogram)
      label_list.append(np.asarray(row['language_encoded']))
    X = np.stack(spectrogram_list, axis=0)
    y = np.stack(label_list, axis=0)
    return X, y
  
  def on_epoch_end(self):
    pass

  def __getitem__(self,index):
    batch_df = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
    X, y = self.__get_batch(batch_df)
    return X, y

  def __len__(self):
    return self.n
