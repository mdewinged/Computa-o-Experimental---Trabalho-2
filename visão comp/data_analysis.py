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
import numpy as np                                                         
import scipy as sp                                                         
import scipy.stats       
import itertools  
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score                                                


# São lidos todos os arquivos de treinamento, validação e de predição
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


acc = []
for res in results:
    acc.append(res[1]['val_accuracy'][99])

acc = np.array(acc)
glob_sum  = sum(acc)
glob_mean = stat.mean(acc)
acc = acc.reshape((5,5))

# Calculates de effect of each column and row
col_sum  = np.array([sum(col) for col in acc])
col_mean = np.array([stat.mean(col) for col in acc])
col_effect = col_mean - glob_mean

row_sum  = np.array([sum(row) for row in acc])
row_mean = np.array([stat.mean(row) for row in acc.T])
row_effect = row_mean - glob_mean

# Calculates the percentage of each column
estimated_val = ((np.zeros((5,5)) + glob_mean + row_effect).T + col_effect).T
err = acc - estimated_val
sse = sum(sum(err**2))

# Compute the total variation and impact of each variation
ssy = sum(sum(acc**2))
ss0 = len(col_effect)*len(row_effect)*(glob_mean**2)
ssa = len(row_effect)*sum(col_effect**2)
ssb = len(col_effect)*sum(row_effect**2)
sst = ssy - ss0
sse = sst - ssa - ssb

a_impact = 100*ssa/sst
b_impact = 100*ssb/sst
unexplained_var = 100*sse/sst 
a_impact + b_impact + unexplained_var


# Analysis of variance
msa = ssa / (len(col_effect) - 1)
msb = ssb / (len(row_effect) - 1)
mse = sse / ((len(col_effect) - 1) * (len(row_effect) - 1))

Fa = msa / mse
Fb = msb / mse

dt = pd.DataFrame({'acc': acc.reshape(25), 'err': err.reshape(25)})
dt = dt.sort_values(by = 'acc')
plt.plot(dt['err'], dt['acc'], 'o')
plt.show()

# Intervalo de confiança
sde = mse**(1/2)
sdu = sde / ((len(row_effect) * len(col_effect))**(1/2))
sda = mse*((len(col_effect)-1)/ (len(col_effect)*len(row_effect)))**(1/2)
sdb = mse*((len(row_effect)-1)/ (len(col_effect)*len(row_effect)))**(1/2)

t_val = 4.014996
ic_u = (glob_mean - sdu*t_val, glob_mean + sdu*t_val)
rangea = (sda*t_val/(((len(col_effect) - 1) * (len(row_effect) - 1))**(1/2)))
ic_a = [(mean_a - rangea, mean_a + rangea)for mean_a in col_mean]
rangeb = (sdb*t_val/(((len(col_effect) - 1) * (len(row_effect) - 1))**(1/2)))
ic_b = [(mean_b - rangea, mean_b + rangea)for mean_b in row_mean]


# Intervalo de Confiança para proporção
t_val = 2.776445 # 95% com 4 graus de lib
sd_c = np.array([stat.stdev(col) for col in acc])
range_c = sd_c*t_val/(len(row_mean)**(1/2))
ic_c_lw = [col - r for col, r in zip(col_mean, range_c)]
ic_c_up = [col + r for col, r in zip(col_mean, range_c)]

cat_name = ['2', '4', '8', '16', '32']
for lower, upper, y in zip(ic_c_lw, ic_c_up, range(len(ic_c_lw))):
    plt.plot((lower,upper),(y,y),'ro-',color='orange')

plt.yticks(range(len(ic_c_lw)),list(cat_name))
plt.xlabel("Classification Accuracy")
plt.ylabel("Kernel Size")
plt.title("Intervalos de Confiança por Kernel Size")
plt.show()


sd_r = np.array([stat.stdev(row) for row in acc.T])
range_r = sd_r*t_val/(len(col_mean)**(1/2))
ic_r_lw = [col - r for col, r in zip(row_mean, range_r)]
ic_r_up = [col + r) for col, r in zip(row_mean, range_r)]

cat_name = ['2', '8', '16', '32', '64']
for lower, upper, y in zip(ic_r_lw, ic_c_up, range(len(ic_r_lw))):
    plt.plot((lower,upper),(y,y),'ro-',color='orange')

plt.yticks(range(len(ic_r_lw)),list(cat_name))
plt.xlabel("Classification Accuracy")
plt.ylabel("Kernel Size")
plt.title("Intervalos de Confiança por Kernel Size")
plt.show()



# Converte um array de pesos para array de labels
def cvt_hot_to_lst(hot_list):
    return np.argmax(hot_list, axis = 1)



# Calcula os scores e as curvas de ROC
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



# Converte todos os arrays de peso para array de labels
def cvt_all(list_structure):
    for i in range(len(list_structure)):
        list_structure[i][1][0] = cvt_hot_to_lst(list_structure[i][1][0])
        list_structure[i][1][1] = cvt_hot_to_lst(list_structure[i][1][1])
    return list_structure

predict = cvt_all(predict)
    

    
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





