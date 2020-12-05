from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import initializers, backend, optimizers
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras import optimizers as op
from keras import backend as K 
from keras import regularizers
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from math import floor 
import cv2 as cv
from glob import glob
import fnmatch
from PIL import Image
import numpy as np
import pandas as pd
import json
import seaborn as sns
import gc


# Leitura das imagens contidas no Dataset
imagePatches = 0
if os.path.isfile('patches.json'):                                                    # Caso o arquivos com todos os patches das imagens exista, apenas ler o arquivo
    f = open('patches.json', 'r')
    text = f.read()
    imagePatches = json.loads(text)
    f.close()
else:                                                                                 # Caso não exista o arquivo dos patches, procurar todos os patches .png do dataset
    imagePatches = glob('dataset/**/**/*.png', recursive=True)
    with open(("patches.json"), 'w', encoding='utf8') as outfile:
        json.dump(imagePatches, outfile, indent = 2, ensure_ascii = False)                  # Cria-se o arquivo de patches 



# Separa os patches segundo suas classes, ou seja, classe 0 é não cancerígena, classe 1 é cancerígena
patternZero = '*class0.png'
patternOne  = '*class1.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne  = fnmatch.filter(imagePatches, patternOne)


# Em seguida é carregadas as imagens para a memória
x = []
y = []
num_samples = 500
img_size = (64, 64)

def proc_img (num_samples):
    global class0, class1
    if num_samples > len(imagePatches): num_samples = len(imagePatches)
    max_size = len(imagePatches)
    prop = len(classOne) / max_size
    [x.append(cv.resize(cv.imread(img), img_size, interpolation=cv.INTER_CUBIC)) for img in classZero[0:floor((1-prop)*num_samples-1)]]
    [y.append(0) for i in range(0,floor((1-prop)*num_samples)-1)]
    [x.append(cv.resize(cv.imread(img), img_size, interpolation=cv.INTER_CUBIC)) for img in classOne[0:floor(prop*num_samples-1)]]
    [y.append(1) for i in range(0,floor(prop*num_samples)-1)]
    

proc_img(num_samples)
print("Proportion of IDC(+) hole dataset: ", len(classOne)/len(imagePatches))



# Prints para verificar se a amostragem continua com a mesma proporção de casos positivos e negativos
df = pd.DataFrame()
df["images"]= x
df["labels"]= y 

print('Total number of images: {}'.format(len(imagePatches)))
print('Number of IDC(-) Images: {}'.format(len(classZero)))
print('Number of IDC(+) Images: {}'.format(len(classOne)))
print("Proportion of IDC(+) hole dataset: ", len(classOne)/len(imagePatches))
print('\nSample size: {}'.format(num_samples))
print('Number of IDC(-) Images on sample: {}'.format(len(df[df.labels == 0])))
print('Number of IDC(+) Images on sample: {}'.format(len(df[df.labels == 1])))
print("Proportion of IDC(+) sample: ", len(df[df.labels == 1])/num_samples)
print('Image shape (Width, Height, Channels): {}'.format(df['images'][0].shape))
del  (imagePatches, classZero, classOne, df)
gc.collect()



# Os dados são dividos para os conjuntos de treinamento e de teste
x = np.array(x)
x = x/255
X_train, X_test, Y_train, Y_test = train_test_split(x, y, shuffle = True, test_size=0.3)

del (x, y)
gc.collect()

print("Proportion of IDC(+) on train: ", len([result for result in Y_train if result == 1])/len(Y_train))
print("Proportion of IDC(+) on test: ", len([result for result in Y_test if result == 1])/len(Y_test))



# Em seguida os vetores de labels são convertidos para vetores de pesos e as imagens são convertidas para serem compatíveis com a rede neural
# Também é feito o balanceamento do dataset, ou seja, a mesma quantidade de casos IDC(+) e IDC(-)
Y_trainHot = to_categorical(Y_train, num_classes = 2)
Y_testHot = to_categorical(Y_test, num_classes = 2)

X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)

ros = RandomUnderSampler(sampling_strategy='auto')
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 2)
Y_testRosHot = to_categorical(Y_testRos, num_classes = 2)

for i in range(len(X_trainRos)):
    height, width, channels = img_size[0],img_size[1],3
    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)

for i in range(len(X_testRos)):
    height, width, channels = img_size[0],img_size[1],3
    X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)

del (Y_trainRos, Y_testRos, X_trainRos, X_testRos, X_trainFlat, X_testFlat, Y_trainHot, Y_testHot)
gc.collect()



# Callback da rede neural
class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# Controla o learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



# Arquitetura da Rede Neural
def build_model(kernel_size, num_filter, input_shape):
    model = Sequential() 
    model.add(Conv2D(filters=num_filter, input_shape=input_shape, kernel_size=kernel_size))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=7, activation = "relu", kernel_size=kernel_size, strides = (128,128)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    opt = optimizers.SGD(learning_rate=0.001, momentum = 0.9)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'mean_absolute_error']) 
    return model



# Configuração do parâmetros do treinamento
kernel_sizes = [(2,2), (4,4), (8,8), (16,16), (32,32)]
num_filters  = [2, 8, 32, 64, 128]
batch_size = 16
steps_per_epoch = 10
epochs = 100



# Gerador para Data Agumentation
datagen = ImageDataGenerator(
    featurewise_center = False,  
    samplewise_center = False,  
    featurewise_std_normalization = False, 
    samplewise_std_normalization = False,  
    zca_whitening = False,
    rotation_range = 20,  
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True, 
    vertical_flip = True)  



# Loop de treinamento da rede neural. Para cada loop, é executado um treinamento com uma configuração específica
for kernel_size in kernel_sizes:
    for num_filter in num_filters:
        backend.clear_session()
        iteration_name = (str(kernel_size[0]) + str("_") + str(num_filter))
        if os.path.isfile(iteration_name +'.json'):
            continue
        model = build_model(kernel_size, num_filter, (64, 64, 3))
        lrs = LearningRateScheduler(scheduler)
        history = model.fit(datagen.flow(X_trainRosReshaped, Y_trainRosHot), 
                            batch_size = batch_size, 
                            steps_per_epoch = steps_per_epoch, 
                            epochs= epochs, 
                            validation_data = (X_testRosReshaped, Y_testRosHot), 
                            callbacks = [MetricsCheckpoint('logs'), lrs])
        history.history['lr'] = [np.float64(num) for num in history.history['lr']]
        with open((iteration_name +".json"), 'w', encoding='utf8') as outfile:
            json.dump(history.history, outfile, indent = 2, ensure_ascii = False)
        predicted_labels = model.predict(X_testRosReshaped)
        pred_results = []
        pred_results.append(predicted_labels.tolist())
        pred_results.append(Y_testRosHot.tolist())
        [print("Predicted X Real label: ", predicted_labels[i], Y_testRosHot[i]) for i in range(len(predicted_labels))]
        with open(("pred_" + iteration_name +".json"), 'w', encoding='utf8') as outfile:
            json.dump(pred_results, outfile, indent = 2, ensure_ascii = False)
        gc.collect()