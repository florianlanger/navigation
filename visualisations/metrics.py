import gensim.downloader as api
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
from datetime import datetime
from torch.utils import data
import fasttext
import fasttext.util
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

#sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation/visualisations"))

from ablation_loss_visualisations import visualise



data = np.load('data.npz')

for i in range(3):
    visualise(data["val_predictions"][i][1:2],data["val_targets"][1:2],'visualisations/model_'+str(i))


#Three things distance from max prediction, relative probabilitiy at target, at RC curve

def average_dist_of_max_prediction(predictions,target_indices):
    max_indices = np.argmax(predictions,axis=1)
    distances = np.zeros(max_indices.shape[0])
    for i in range(max_indices.shape[0]):
        distances[i] = calc_dist_indices(max_indices[i],target_indices[i])
    return np.mean(distances)

def relative_probability(predictions,target_indices):
    probabilities = np.zeros(predictions.shape[0])
    for i in range(predictions.shape[0]):
        probabilities[i] = predictions[i,int(target_indices[i])] / np.max(predictions[i])
    return np.mean(probabilities)

#Find relative probability
# relative_probs = np.empty((3,50))
# for i in range(3):
#     for j in range(50):
#         target_position = data["val_targets"][j]

#         indices = 2 * target_position + 4
#         target_index = int(indices[0] * 81 + indices[1]*9 + indices[2])

#         relative_probs[i,j] = data["val_predictions"][i,j,target_index] / np.max(data["val_predictions"][i,j])
   
# print(relative_probs)
# print(np.mean(relative_probs,axis=1))


def calc_min_dist_threshold(sorted_predictions,target_indices,threshold):
    min_dist = 100*np.ones(sorted_predictions.shape[0])
    for k in range(sorted_predictions.shape[0]):
        sorted_predictions_single_example_descending = sorted_predictions[k][::-1]
        for i in range(threshold):
            distance = calc_dist_indices(target_indices[k],sorted_predictions_single_example_descending[i])
            if distance < min_dist[k]:
                min_dist[k] = distance
    return min_dist

def calc_dist_indices(index_1,index_2):
    distance = np.linalg.norm(index_to_position(index_1) - index_to_position(index_2))
    return distance

def index_to_position(index):
    indices = np.array([index //81, (index// 9) % 9, index % 9])
    indices -= 4
    position = indices * np.ones(3)/2
    return position

def plot_rc_curve(distances,start,stop,steps,label):
    x = np.arange(start,stop,steps)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,distances[0],label="Learn embedding + average")
    ax.plot(x,distances[1],label="Learn embedding + LSTM") 
    ax.plot(x,distances[2],label="Fasttext embedding + norm average")
    ax.plot(x,distances[3],label="Fasttext embedding + LSTM")
    ax.legend()

    ax.set_xlabel('n max predictions')

    ax.set_ylabel('Distance to target (m)')

    fig.savefig('roc_curves/all_architecture_{}.png'.format(label),dpi=250,bbox_inches = "tight")
    plt.close(fig)

start = 1
stop = 201
steps = 10
target_indices = np.load('target_indices.npz')


# for i in tqdm(range(1,5)):
#     data = np.load('data_architectures/architecture_{}.npz'.format(str(i).zfill(2)))

#     distances = np.empty((2,int((stop-start)/steps)))

#     for h,kind in enumerate(["train","val"]):
#         #for i in tqdm(range(3)):
#         sorted_predictions = np.argsort(data[kind+"_predictions"],axis=1)
#         for j,threshold in enumerate(range(start,stop,steps)):
#             min_distances = calc_min_dist_threshold(sorted_predictions,target_indices[kind],threshold)
#             distances[h,j] = np.mean(min_distances)


#     np.savez('data_architectures/roc_curve_architecture_{}.npz'.format(str(i).zfill(2)),distances=distances)



#ROC curves:
# all_data = np.zeros((4,2,int((stop-start)/steps)))

# for i in range(1,5):
#     data = np.load('data_architectures/roc_curve_architecture_{}.npz'.format(str(i).zfill(2)))
#     all_data[i-1] = data["distances"]

# plot_rc_curve(all_data[:,0,:],start,stop,steps,"train")
# plot_rc_curve(all_data[:,1,:],start,stop,steps,"val")


# Dist to max 
# all_train_predictions = np.zeros((4,200,729))
# all_val_predictions = np.zeros((4,50,729))
# for i in tqdm(range(1,5)):
#     data = np.load('data_architectures/architecture_{}.npz'.format(str(i).zfill(2)))
#     print('------- model 1 ------')
#     print('train: ',relative_probability(data["train_predictions"],target_indices["train"]))
#     print('val: ',relative_probability(data["val_predictions"],target_indices["val"]))

# for i in range(3):
#     print('train: ',average_dist_of_max_prediction(data["train_predictions"][i],target_indices["train"]))
#     print('val: ',average_dist_of_max_prediction(data["val_predictions"][i],target_indices["val"]))
#     print('----')



#     data = np.load('data.npz')

#      for i in range(3):
#         visualise(data["val_predictions"][i],data["val_targets"],'model_'+str(i))

#     distances = np.empty((2,int((stop-start)/steps)))


#         sorted_predictions = np.argsort(data["val_predictions"][i],axis=1)
#         for j,threshold in enumerate(range(start,stop,steps)):
#             min_distances = calc_min_dist_threshold(sorted_predictions,data["val_targets"][i],threshold)
#             distances[h,j] = np.mean(min_distances)


#     np.savez('data_architectures/roc_curve_architecture_{}.npz'.format(str(i).zfill(2)),distances=distances)

# data = np.load('roc_curve_distances.npz')
# for k in data.files:
#     print(k)
# print(data["distances"].shape)