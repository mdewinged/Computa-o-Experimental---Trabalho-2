import statistics as stat
from matplotlib import pyplot as plt
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
from time import time
import numpy as np                                                         
import scipy as sp                                                         
import scipy.stats       
import itertools                                                  

kernel_sizes = [(2,2), (4,4), (8,8), (16,16), (32,32)] 
num_filters  = [2, 8, 32, 64, 128]
        
results = []
predict = []
for kernel_size in kernel_sizes:
    for num_filter in num_filters:
        iteration_name = (str(kernel_size[0]) + str("_") + str(num_filter))
        if os.path.isfile(iteration_name + str(".json")):
            f = open(iteration_name + str(".json"), 'r')
            text = f.read()
            data = []
            data.append(iteration_name)
            data.append(json.loads(text))
            results.append(data)
            f.close()
        else: 
            print("Couldn't find ", iteration_name, " file\n")
        
        if os.path.isfile("pred_" + iteration_name + str(".json")):
            f = open("pred_" + iteration_name + str(".json"), 'r')
            text = f.read()
            data = []
            data.append("pred_" + iteration_name)
            data.append(json.loads(text))
            predict.append(data)
            f.close()
        else: 
            print("Couldn't find ", ("pred_" + iteration_name + str(".json")), " file\n")


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def cvt_hot_to_lst(hot_list):
    return np.argmax(hot_list, axis = 1)


def roc_eval(labels, name):
    ytest = np.expand_dims(cvt_hot_to_lst(labels[1]), axis = 1)
    ns_probs = [0 for _ in range(len(ytest))]
    lr_probs = np.expand_dims([pred[1] for pred in labels[0]], axis = 1)
    ns_auc = roc_auc_score(ytest, ns_probs)
    lr_auc = roc_auc_score(ytest, lr_probs)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
    #plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label= name)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.title("ROC Curve por Número de Filtros 8")
    return lr_auc

roc_result = []
for pred in predict:
    aux = roc_eval(pred[1], pred[0])
    roc_result.append([pred[0], aux])

aux = roc_eval(predict[20][1], predict[20][0])
aux = roc_eval(predict[21][1], predict[21][0])
aux = roc_eval(predict[22][1], predict[22][0])
aux = roc_eval(predict[23][1], predict[23][0])
aux = roc_eval(predict[24][1], predict[24][0])
plt.show()

aux = roc_eval(predict[1][1], predict[1][0])
aux = roc_eval(predict[6][1], predict[6][0])
aux = roc_eval(predict[11][1], predict[11][0])
aux = roc_eval(predict[16][1], predict[16][0])
aux = roc_eval(predict[21][1], predict[21][0])
plt.show()

with open(("roc_result.json"), 'w', encoding='utf8') as outfile:
        json.dump(roc_result, outfile, indent = 2, ensure_ascii = False)


def cvt_all(list_structure):
    for i in range(len(list_structure)):
        list_structure[i][1][0] = cvt_hot_to_lst(list_structure[i][1][0])
        list_structure[i][1][1] = cvt_hot_to_lst(list_structure[i][1][1])
    return list_structure


predict = cvt_all(predict)



def count_precision_recall(list1, list2):
    false_pos = 0
    false_neg = 0
    correct = 0
    for l1,l2 in zip(list1, list2):
        if(l1==l2):
            correct +=1
        elif(l2==0):
            false_neg +=1
        elif(l2==1):
            false_pos +=1
    return(len(list1), correct, false_neg, false_pos)


for pred in predict:
    total, a, b, c = count_precision_recall(pred[1][0], pred[1][1])
    print(pred[0], ":\t precision ", a/total, "\tfalseneg ", b/total, "\tfalsepos ", c/total)

        



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(np.expand_dims(predict[0][1][1], axis = 0), np.expand_dims(predict[0][1][0], axis=0))


# Plots de verificação se cada experimento estava consistente entre validação e treinamento
for i in range(len(results)):
    plt.plot(results[i][1]['val_accuracy'], label = "val")
    plt.plot(results[i][1]['accuracy'], label = "train")
    plt.xlabel('n_epochs')
    plt.ylabel('accuracy')
    plt.title("Model Accuracy: " + str(results[i][0]))
    plt.legend()
    plt.show()
    plt.plot(results[i][1]['val_loss'], label = "val")
    plt.plot(results[i][1]['loss'], label = "train")
    plt.xlabel('n_epochs')
    plt.ylabel('loss')
    plt.title("Model loss: " + str(results[i][0]))
    plt.legend()
    plt.show()


#################################################### Plot de kernel size  ####################################################################
# Plot de todos os gráficos segundo o tamanho do filtro
kernels = []
for res in results:
    kernels.extend(res[1]['val_accuracy'])

kernels = np.expand_dims(kernels, axis = 1)

kernel2 = kernels[0:499]
kernel4 = kernels[500:999]
kernel8 = kernels[1000:1499]
kernel16 = kernels[1500:1999]
kernel32 = kernels[2000:2499]

plt.plot(kernel2, label = "kernel size 2")
plt.plot(kernel4, label = "kernel size 4")
plt.plot(kernel8, label = "kernel size 8")
plt.plot(kernel16, label = "kernel size 16")
plt.plot(kernel32, label = "kernel size 32")
plt.xlabel("número de epochs acumulado")
plt.ylabel("accuracy")
axes = plt.gca()
axes.set_ylim([0.5,0.82])
plt.title("Model Accuracy vs Epochs\nkernel size & number of filters: " + str(kernel_sizes[int((i/5))][0]))
plt.legend()
plt.show()
#################################################### Plot de kernel size  ####################################################################


#################################################### Plot de acurácia de num de filtros e tamanho do filtro ###############################################################
# Plot de todos os gráficos segundo o tamanho do filtro
for i in range(len(results)):
    plt.plot(results[i][1]['val_accuracy'], label = results[i][0])
    if(((i+1) % 5 == 0) & (i != 0)):
        plt.xlabel('n_epochs')
        plt.ylabel('accuracy')
        plt.title("Model Accuracy vs Epochs\nkernel size: " + str(kernel_sizes[int((i/5))][0]))
        plt.legend()
        plt.savefig("kernel_sz_" + str(kernel_sizes[int((i/5))][0]) + ".png")
        plt.show()


# Plot de todos os gráficos segundo a quantidade de filtros
for j in range(5):
    for i in range(len(results)):
        if((i+1) % 5 == (j+1)%5):
            plt.plot(results[i][1]['val_accuracy'], label = results[i][0])
    plt.xlabel('n_epochs')
    plt.ylabel('accuracy')
    plt.title("Model Accuracy vs Epochs\nnum_filters: " + str(num_filters[int((j))]))
    plt.legend()
    plt.savefig("num_filters_" + str(num_filters[int(j)]) + ".png")
    plt.show()
#################################################### Plot de acurácia de num de filtros e tamanho do filtro ###############################################################



#################################################### Plot de kernel size  ####################################################################
# Plot de todos os gráficos segundo o tamanho do filtro
for i in range(len(results)):
    plt.plot(results[i][1]['val_mean_absolute_error'], label = results[i][0])
    if(((i+1) % 5 == 0) & (i != 0)):
        plt.xlabel('n_epochs')
        plt.ylabel('accuracy')
        plt.title("Model Mean Abs Err vs Epochs\nkernel size: " + str(kernel_sizes[int(i/5)][0]))
        plt.legend()
        plt.savefig("erro_kernel_sz_" + str(kernel_sizes[int((i/5))][0]) + ".png")
        plt.show()


# Plot de todos os gráficos segundo a quantidade de filtros
for j in range(5):
    for i in range(len(results)):
        if((i+1) % 5 == (j+1)%5):
            plt.plot(results[i][1]['val_mean_absolute_error'], label = results[i][0])
    plt.xlabel('n_epochs')
    plt.ylabel('accuracy')
    plt.title("Model Mean Abs Err vs Epochs\nnum_filters: " + str(num_filters[int(j)]))
    plt.legend()
    plt.savefig("erro_num_filters_" + str(num_filters[int(j)]) + ".png")
    plt.show()
#################################################### Plot de kernel size  ####################################################################





