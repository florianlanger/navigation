import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import shutil
from absl import flags
from absl import app
from tqdm import tqdm
from datetime import datetime

from losses.basic_losses import calc_individual_loss, classify_predictions, calc_individual_loss_dot_product, calc_individual_loss_kl_div, calc_target_distribution
from data.data_loader import Sampler,Dataset, load_data_set
from utilities import load_config, load_model, update_accuracies, update_checkpoint_epochs, update_counters_actions, create_directories, write_to_file
from plots import plot_frequencies, plot_history, visualise_pairs, plot_heatmap_accuracies

FLAGS = flags.FLAGS
flags.DEFINE_string("weights", None, 'Path to a model without "_model.pth" e.g. "exp_1/checkpoints/best_train"')


def train(network, config, train_loader, optimizer, epoch, exp_path, min_train_loss, heatmap_accuracies,counters_actions):
    network.train()
    training_loss = 0.
    accuracy = 0.
    bs = config['training']['batch_size']
    indices = torch.empty(config['training']['pairs_per_epoch'],2).cuda()
    for batch_idx, (data,positions,index,action_penalties) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        individual_loss = calc_individual_loss(output,action_penalties,config).view(-1)
        loss = torch.mean(individual_loss)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
        with torch.no_grad():
            boolean_predictions = classify_predictions(output,action_penalties)
            update_accuracies(boolean_predictions,positions,heatmap_accuracies,config)
            acc_mean = torch.sum(boolean_predictions)/float(bs)
            accuracy = 1/(batch_idx + 1) * acc_mean + (batch_idx/(batch_idx+1)) * accuracy
            update_counters_actions(action_penalties,output,counters_actions)
            indices[batch_idx*bs:(batch_idx+1)*bs] = index

        # after 'log_interval' iterations write to log file
        if batch_idx % config["training"]["log_interval"] == 0:
        # all losses are per training example
            line = 'Epoch: {} Average Training Loss: {:.4f} Accuracy: {:.4f} Loss Last Batch: {:.4f}'.format(
        epoch,training_loss/((batch_idx +1)), accuracy,loss.item())
            write_to_file(exp_path +'/log.txt','\n'+line)

            print(line, end="\r", flush=True)

        #produce visualisation of first images in first batch every few epochs
        if batch_idx == 0 and (epoch-1) % config["visualisations"]["interval"] == 0:
           with torch.no_grad():
               path = '{}/visualisations/single_pairs/epoch_{}'.format(exp_path,epoch)
               os.mkdir(path)
               visualise_pairs(data,positions,output,action_penalties,path,config,
               individual_loss,str(config["loss"]["type"]))
    print('\n')
    # This is only precise if have full number of batches per dataset
    training_loss /= (batch_idx + 1)

    if epoch == 1:
        write_to_file(exp_path + "/history.csv",'epoch,training_loss,training_accuracy\n')
        write_to_file(exp_path + "/checkpoints/model_info.txt",' \n ')

    if (epoch-1) % config["training"]["save_interval"] == 0:
        torch.save(network.state_dict(), exp_path + '/checkpoints/epoch_{}_model.pth'.format(epoch))
        torch.save(train_loader, exp_path + '/checkpoints/epoch_{}_data_loader.pth'.format(epoch))


    write_to_file(exp_path + '/history.csv',"{},{},{}\n".format(epoch,training_loss,accuracy))
    torch.save(network.state_dict(), exp_path + '/checkpoints/last_epoch_model.pth')
    torch.save(train_loader, exp_path + '/checkpoints/last_epoch_data_loader.pth')
    if training_loss < min_train_loss:
        torch.save(network.state_dict(), exp_path + '/checkpoints/best_train_model.pth')
        torch.save(train_loader, exp_path + '/checkpoints/best_train_data_loader.pth')
        update_checkpoint_epochs(epoch, training_loss,'train',exp_path)
        return training_loss, indices
    else:
        return min_train_loss, indices


def find_hard_pairs(network,config,train_loader,dataset,epoch, exp_path):
    print('Recalculate losses to find hard pairs')
    with torch.no_grad():
        bs = config['training']['batch_size']
        losses = torch.empty(config['training']['pairs_per_epoch']).cuda()
        indices = torch.empty(config['training']['pairs_per_epoch'],2,dtype=torch.int64).cuda()
        predictions = torch.empty(config['training']['pairs_per_epoch'],config["model"]["number_outputs"]).cuda()
        all_positions = torch.empty(config['training']['pairs_per_epoch'],2*config["data"]["dim_position"]).cuda()
        all_action_penalties = torch.empty(config['training']['pairs_per_epoch'],config["model"]["number_outputs"]).cuda()
        for batch_idx, (data, positions,index,action_penalties) in enumerate(train_loader):
            output = network(data)
            individual_loss = calc_individual_loss(output,action_penalties,config).view(-1)

            losses[batch_idx*bs:(batch_idx+1)*bs] = individual_loss.view(-1)
            indices[batch_idx*bs:(batch_idx+1)*bs] = index
            predictions[batch_idx*bs:(batch_idx+1)*bs] = output
            all_positions[batch_idx*bs:(batch_idx+1)*bs] = positions
            all_action_penalties[batch_idx*bs:(batch_idx+1)*bs] = action_penalties

        sorted_losses, sorted_loss_indices = torch.sort(losses,descending=True)
        sorted_indices, sorted_predictions = indices[sorted_loss_indices], predictions[sorted_loss_indices]
        sorted_positions, sorted_action_penalties = all_positions[sorted_loss_indices],all_action_penalties[sorted_loss_indices]

        #produce visualisation
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            number_hard_pairs = config["visualisations"]["hard_pairs"]
            path = '{}/visualisations/hard_pairs/epoch_{}'.format(exp_path,epoch)
            os.mkdir(path)

            all_images = torch.empty(number_hard_pairs,2,3,100,100).cuda()
            with open('{}/overview.txt'.format(path),'w') as f:
                for i in range(number_hard_pairs):
                    # write to overview text file
                    pos_1 = list(sorted_positions[i,0:int(positions.shape[1]/2)].cpu().numpy())
                    pos_2 = list(sorted_positions[i,int(positions.shape[1]/2):].cpu().numpy())
                    f.write('Current: {} Goal: {} : Loss: {}\n'.format(pos_1,pos_2,sorted_losses[i].item()))
                    # get corresponding images 
                    images,_,_,_ = dataset[sorted_indices[i].cpu()]
                    all_images[i] = images

            visualise_pairs(all_images,sorted_positions[:number_hard_pairs],sorted_predictions[:number_hard_pairs],
                        sorted_action_penalties[:number_hard_pairs],path,config,sorted_losses[:number_hard_pairs],str(config["loss"]["type"]),kind="hard_pairs")

        return sorted_indices


def main(argv):
    del(argv)

    # get absolute path to current file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load template config
    config = load_config('{}/../config.json'.format(dir_path))
    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"],
        datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    if os.path.exists(exp_path):
        raise Exception("An experiment named {} already exists. Please use a different name.".format(config["experiment_name"]))

    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    # load dataset
    dataset = load_data_set(config)

    # initialise model and optimiser
    network = load_model(config)
    network = nn.DataParallel(network)
    network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=config["training"]["learning_rate"])


    if FLAGS.weights is None:
        sampler = Sampler(config)
        train_loader =  torch.utils.data.DataLoader(dataset, batch_size = config["training"]["batch_size"], sampler=sampler)
    else:
        write_to_file(exp_path+'/log.txt','\n Load weights from: ' + FLAGS.weights +'\n')
        print('Load weights from ' + FLAGS.weights)
        network.load_state_dict(torch.load('{}/../experiments/{}_model.pth'.format(dir_path, FLAGS.weights)))
        
        if os.path.isfile('{}/../experiments/{}_data_loader.pth'.format(dir_path, FLAGS.weights)):
            print('Load train loader from ' + FLAGS.weights)
            train_loader = torch.load('{}/../experiments/{}_data_loader.pth'.format(dir_path, FLAGS.weights))
        else:
            sampler = Sampler(config)
            train_loader =  torch.utils.data.DataLoader(dataset, batch_size = config["training"]["batch_size"], sampler=sampler)


    min_train_loss = np.inf
    epochs = range(1,config["training"]["n_epochs"]+1)
    for epoch in epochs:
        if "place" in config["data"]:
            number_states = config["data"]["states_each_dimension"]
            heatmap_mean = torch.zeros(2,number_states[0],number_states[1],number_states[2],number_states[3],1).cuda()
            heatmap_counter = torch.ones(2,number_states[0],number_states[1],number_states[2],number_states[3],1).cuda()
            heatmap_counter[heatmap_counter!=0] = float('nan')
            heatmap_accuracies = torch.cat((heatmap_mean,heatmap_counter),dim=5)

        else:
            heatmap_accuracies = torch.zeros(2,20,20,6,4,2).cuda()
        #First row is predictions (only one prediction per example), second row is ground truth (can be multiple)
        counters_actions = torch.zeros(2,config["model"]["number_outputs"]).cuda()
        min_train_loss, indices = train(network,config,train_loader,optimizer,epoch, exp_path, min_train_loss, heatmap_accuracies, counters_actions)
        if config["sampler"]["ratio_hard_pairs"] != 0.0:
            hard_pairs = find_hard_pairs(network,config,train_loader,dataset,epoch, exp_path)
            train_loader.sampler.update_sampler(hard_pairs)

        else:
            train_loader.sampler.random()

        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_heatmap_accuracies('{}/visualisations/heatmaps'.format(exp_path), epoch, heatmap_accuracies.cpu(),config)
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)
            plot_frequencies(counters_actions, epoch, exp_path, config)

if __name__ == "__main__":
    app.run(main)