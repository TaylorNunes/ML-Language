import tensorflow as tf
import keras_tuner as kt
import generator as my_generator

# Hyperparameter tuner class
class MyHyperModel(kt.HyperModel):
  def __init__(self, train_df, validation_df, output_classes, clip_length_seconds):
    self.output_classes = output_classes
    self.clip_length_seconds = clip_length_seconds
    self.train_df = train_df
    self.validation_df = validation_df
    #self.train_generator = None
    #self.validation_generator = None
    
  def build(self, hp):
    
    # Tuned parameters
    learning_rate = 10**hp.Float('learning_rate_exp', min_value=-4, max_value=-3, step=1)
    #batch_size = hp.Int('batch_size', min_value=1, max_value=8, step=1)
    batch_size = 4
    spectrogram_nfft = 2**hp.Int('nfft_exp', min_value=8, max_value=10, step=1)
    spectrogram_window = 2**hp.Int('window_exp', min_value=6, max_value=8, step=1)
    spectrogram_stride = 2**5
    spectrogram_mels = hp.Int('mels', min_value=256, max_value=512, step=20)


    self.train_generator = my_generator.MyGenerator(self.train_df, 
                                                batch_size, 
                                                self.clip_length_seconds, 
                                                spec_nfft=spectrogram_nfft,
                                                spec_window=spectrogram_window,
                                                spec_stride=spectrogram_stride,
                                                spec_mels=spectrogram_mels)
    self.validation_generator = my_generator.MyGenerator(self.validation_df,
                                                    batch_size, 
                                                    self.clip_length_seconds, 
                                                    spec_nfft=spectrogram_nfft,
                                                    spec_window=spectrogram_window,
                                                    spec_stride=spectrogram_stride,
                                                    spec_mels=spectrogram_mels)
    image_size = self.train_generator.spectrogram_shape
    output_neurons = self.output_classes
    if output_neurons == 2:
      output_neurons = 1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,3, input_shape=image_size, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(32,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(64,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(128,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(256,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(512,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(output_neurons, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()
    return model

  def fit(self, hp, model, **kwargs):
    print('TG: ', self.train_generator)
    return model.fit(self.train_generator, validation_data=self.validation_generator, **kwargs)

# Standard model class
class MyModeler():
  def __init__(self,
                input_shape,
                output_classes):
    self.output_classes = output_classes
    self.input_shape = input_shape
  
  def make_model(self):
    output_neurons = self.output_classes
    if self.output_classes == 2:
      output_neurons = 1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,3, input_shape=self.input_shape, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(32,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(64,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(128,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(256,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(512,3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(output_neurons, activation='sigmoid'))
    return model
