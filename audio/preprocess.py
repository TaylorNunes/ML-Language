import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import os
import random

class ShortAudioPreprocesser(): # Called by the generator
  def __init__(self, 
                output_length, 
                output_rate,
                spec_nfft,
                spec_window,
                spec_stride,
                spec_mels):
    self.output_length = output_length
    self.output_rate = output_rate
    self.output_length_samples = self.output_length * self.output_rate
    self.spec_nfft = spec_nfft
    self.sepc_window = spec_window
    self.spec_stride = spec_stride
    self.spec_mels = spec_mels

  def get_standard_audio(self, file_path):
    audio = tfio.audio.AudioIOTensor(file_path)
    audio = self.__change_rate(audio, self.output_rate)
    audio = self.__random_padding(audio, self.output_length_samples)
    audio = self.__random_slice(audio, self.output_length_samples)
    audio = self.__convert_mono(audio)
    audio = self.__normalize_height(audio.numpy())
    return audio

  def get_spectrogram(self, audio):
    tensor = tf.cast(audio, tf.float32)
    tensor = tf.squeeze(tensor)
    spectrogram = tfio.audio.spectrogram(tensor, nfft=self.spec_nfft, window=self.sepc_window, stride=self.spec_stride)
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=self.output_rate, mels=self.spec_mels, fmin=0, fmax=8000)
    db_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
    return db_mel_spectrogram

  def __change_rate(self, audio, new_rate):
    rate = tf.cast(audio.rate, tf.int64)
    if rate == new_rate:
      return audio
    return tfio.audio.resample(audio.to_tensor(), rate, new_rate)
    
  def __random_padding(self, audio, new_length): # must be changed to the proper rate first
    old_length = audio.shape[0]
    if old_length >= new_length:
      return audio
    padding_length = new_length - old_length
    head_padding = np.random.randint(padding_length)
    tail_padding = padding_length - head_padding
    return tf.pad(audio, [[head_padding,tail_padding],[0,0]], 'CONSTANT')
  
  def __random_slice(self, audio, new_length):
    old_length = audio.shape[0]
    if old_length <= new_length:
      return audio
    start =  np.random.randint(old_length - new_length)
    return audio[start:start+new_length]

  def __convert_mono(self, audio):
    if audio.shape[1] == 1:
      return audio
    else:
      return audio.sum(axis=1) / 2
  
  def __normalize_height(self, audio):
    if not audio.max() == audio.min():
      audio = (audio - audio.min()) / (audio.max() - audio.min())  
    else:
      audio = np.ones(audio.shape)
    return audio

class DatasetMaker(): # Called by the training script
  def __init__(self, clips_number):
    self.clips_number = clips_number

  def make_sample_dataframe(self, languages, language_dir_list):
    df_list = []
    for i_lang in range(len(languages)):
      path_list = get_file_list(language_dir_list[i_lang], self.clips_number)
      if len(path_list) < self.clips_number:
        print('Warning, not enough media. {} requested and only {} found'.format(self.clips_number,len(path_list)))
      for file_path in path_list:
        df_list.append(pd.DataFrame({'file_path':file_path,
                                      'language':languages[i_lang],
                                      'language_encoded':i_lang}, index=[0])) 
    df = pd.concat(df_list)
    df = df.sample(frac=1).reset_index(drop=True)
    self.sample_df = df
    return df

  def split_dataframe(self, df, test_ratio):
    test_sample_start = round(len(df)*(1-test_ratio))
    train_df = df.iloc[:test_sample_start]
    test_df = df.iloc[test_sample_start:]
    return train_df, test_df
  
class AudiobookPreprocess(): # Not called, old class for audiobooks
  def __init__(self):
    return 

  def get_train_val_test(self, number_of_samples, test_ratio, validation_ratio, clip_length_seconds):
    number_of_clips = number_of_samples
    total_required_length = clip_length_seconds*number_of_clips
    japanese_file_list = self.get_file_list('audio/japanese')
    english_file_list = self.get_file_list('audio/english')
    japanese_meta_df = self.get_list_metadata(japanese_file_list)
    english_meta_df = self.get_list_metadata(english_file_list)

    # Checks if there is enough media
    total_japanese_length = japanese_meta_df['total_length_seconds'].sum()
    total_english_length = english_meta_df['total_length_seconds'].sum()

    if total_japanese_length < total_required_length:
      print('Not enough japanese media. Requested {} seconds but only {} seconds are available.'.format(
        total_required_length, total_japanese_length))
      return None
    if total_english_length < total_required_length:
      print('Not enough english media. Requested {} seconds but only {} seconds are available.'.format(
        total_required_length, total_english_length))
      return None

    japanese_sample_df = self.get_sample_df(japanese_meta_df, number_of_clips, clip_length_seconds, 0)
    english_sample_df = self.get_sample_df(english_meta_df, number_of_clips, clip_length_seconds, 1)


    df = pd.concat((japanese_sample_df, english_sample_df))
    df = df.sample(frac=1).reset_index(drop=True)
    test_sample_start = round(len(df)*(1-test_ratio))
    validation_sample_start = round(test_sample_start*(1-validation_ratio))
    train_df = df.iloc[:validation_sample_start]
    validation_df = df.iloc[validation_sample_start:test_sample_start]
    test_df = df.iloc[test_sample_start:]

    return train_df, validation_df, test_df

  def get_sample_df(df, number_of_clips, clip_length_seconds,language):
    df_list = [] # list of samples that will be concatinated
    number_of_files = df.shape[0]
    random_file_index_list = np.random.permutation(number_of_files) # list of indicies of files in random order
    sample_index_lists = [] # list of lists where each list contains sample indicies in random order
    for index in range(len(random_file_index_list)):
      file_df = df.iloc[index]    
      max_clips = int(file_df['total_length_seconds'] // clip_length_seconds)
      sample_index_lists.append(np.random.permutation(max_clips))
    
    sample_counter = 0
    while len(df_list) < number_of_clips:
      file_index = sample_counter % number_of_files
      clip_index = sample_counter // number_of_files
      if clip_index >= len(sample_index_lists[random_file_index_list[file_index]]): # Skips files that run out of audio
        sample_counter+=1
        continue
      file_df = df.iloc[random_file_index_list[file_index]]
      time_start = sample_index_lists[random_file_index_list[file_index]][clip_index] * clip_length_seconds
      sample_counter+=1

      df_list.append(pd.DataFrame({'file_path':file_df['file_path'],
                                    'time_start':time_start, 
                                    'clip_length':clip_length_seconds,
                                    'sampling_rate':int(file_df['sampling_rate']), 
                                    'channels':int(file_df['channels']),
                                    'language_encoded':language}, index=[0])) 
    return pd.concat(df_list)

  def get_list_metadata(list):
    df_list = []
    for file in list:
      audio_file = tfio.audio.AudioIOTensor(file)
      sampling_rate = tf.cast(audio_file.rate, tf.int64)
      length_in_samples = audio_file.shape[0]
      length_in_seconds =  length_in_samples / sampling_rate
      channels = int(audio_file.shape[1])
      if channels == 0: channels = 1
      df_list.append(pd.DataFrame({'file_path':file, 
                                    'total_length_samples':length_in_samples, 
                                    'total_length_seconds':length_in_seconds,
                                    'sampling_rate':sampling_rate,
                                    'channels':channels}, index=[0]))
    return pd.concat(df_list)

def get_file_list(base_dir, limit_max):
  list_of_files = os.listdir(base_dir)
  all_files = []
  for item in list_of_files:
      full_path = os.path.join(base_dir, item)
      if os.path.isdir(full_path):
          all_files = all_files + get_file_list(full_path)
      else:
          all_files.append(full_path)
          if len(all_files) >= limit_max:
            break
  return all_files
