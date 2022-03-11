# Audio classification for English and Japanese
A project for classifying speech audio into Japanese and English using a deep neural network

### Installation
This project requires the following packages:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](http://matplotlib.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Tensorflow IO](https://www.tensorflow.org/io/)

These can be installed from the requirements.txt file.

Note - If you have a GPU, install the tensorflow version that supports  
``` bash
pip install -r requirements.txt
```
This project also provides an interactive workbook to train the model. You will need some method to run an interactive notebook, such as [Jupyter](https://jupyter.org/) or the [Jupyter extension](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for visual studio code.

### Usage
To use this project open the interactive notebook ```main.ipynb```. From there you can run the script interactively. you will need to add folders to the audio folder to store the english and japanese audio
```bash
mkdir audio/english
mkdir audio/japanese
```
For my study I used the the [commonvoice](https://commonvoice.mozilla.org/) corpus, an open source voice corpus that contains many languages.

If you want to view the results of the training in an interactive way, you can use [Tensorboard](). The output for tensorboard is currently directed to ```logs/fit```.

### Results 
The deep learning model was trained on 8000 samples for each language resulting in a classification accuracy of 95%. 
