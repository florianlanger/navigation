import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import shutil
from datetime import datetime

from losses.basic_losses import calc_individual_loss, classify_predictions
from utilities import load_config, load_model,load_data_set_and_sampler, update_counters_actions, create_directories, write_to_file
from visualisations.plots import plot_frequencies, plot_history, visualise_action_predictions, plot_heatmap_accuracies
from models.decoder import Decoder_Basic


def train(network, config, train_loader, optimizer, epoch, exp_path,counters_actions):
    network.train()
    training_loss = 0.
    accuracy = 0.
    bs = config['training']['batch_size']
    for batch_idx, (positions,indices,target_distribution) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(positions)
        individual_loss = calc_individual_loss(output,target_distribution)
        loss = torch.mean(individual_loss)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
        with torch.no_grad():
            boolean_predictions = classify_predictions(output,target_distribution)
            acc_mean = torch.sum(boolean_predictions)/float(bs)
            accuracy = 1/(batch_idx + 1) * acc_mean + (batch_idx/(batch_idx+1)) * accuracy
            update_counters_actions(target_distribution,output,counters_actions)

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
                visualise_action_predictions(positions,target_distribution,output,path, config, losses=individual_loss)
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


def find_hard_pairs(network,config,train_loader,dataset,epoch, exp_path):
    print('Recalculate losses to find hard pairs')
    with torch.no_grad():
        bs = config['training']['batch_size']
        all_losses = torch.empty(config['training']['pairs_per_epoch']).cuda()
        all_indices = torch.empty(config['training']['pairs_per_epoch'],2,dtype=torch.int64).cuda()
        all_predictions = torch.empty(config['training']['pairs_per_epoch'],10).cuda()
        all_positions = torch.empty(config['training']['pairs_per_epoch'],8).cuda()
        all_target_distributions = torch.empty(config['training']['pairs_per_epoch'],10).cuda()

        for batch_idx, (positions,indices,target_distribution) in enumerate(train_loader):
            predictions = network(positions)
            individual_loss = calc_individual_loss(predictions,target_distribution)

            all_losses[batch_idx*bs:(batch_idx+1)*bs] = individual_loss.view(-1)
            all_indices[batch_idx*bs:(batch_idx+1)*bs] = indices
            all_predictions[batch_idx*bs:(batch_idx+1)*bs] = predictions
            all_positions[batch_idx*bs:(batch_idx+1)*bs] = positions
            all_target_distributions[batch_idx*bs:(batch_idx+1)*bs] = target_distribution

        sorted_losses, sorted_loss_indices = torch.sort(all_losses,descending=True)
        sorted_indices, sorted_predictions = all_indices[sorted_loss_indices], all_predictions[sorted_loss_indices]
        sorted_positions,sorted_target_distributions = all_positions[sorted_loss_indices],all_target_distributions[sorted_loss_indices]

        #produce visualisation
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            n = config["visualisations"]["hard_pairs"]
            path = '{}/visualisations/hard_pairs/epoch_{}'.format(exp_path,epoch)
            os.mkdir(path)
            visualise_action_predictions(sorted_positions[:n],sorted_target_distributions[:n],sorted_predictions[:n],path, config, losses=sorted_losses[:n])

        return sorted_indices


def main():
    # get absolute path to current file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = json.load(open('{}/../config.json'.format(dir_path),'r'))["gpu"]
    # load template config
    config = load_config('{}/../config.json'.format(dir_path))
    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"], datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)
    # load dataset
    dataset,sampler = load_data_set_and_sampler(config,exp_path)

    # initialise model and optimiser
    network = load_model(config)
    network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=config["training"]["learning_rate"])

    train_loader =  torch.utils.data.DataLoader(dataset, batch_size = config["training"]["batch_size"], sampler=sampler)

    epochs = range(1,config["training"]["n_epochs"]+1)
    for epoch in epochs:
        counters_actions = torch.zeros(2,10).cuda()
        train(network,config,train_loader,optimizer,epoch, exp_path,counters_actions)
        if config["sampler"]["ratio_hard_pairs"] != 0.0:
            hard_pairs = find_hard_pairs(network,config,train_loader,dataset,epoch, exp_path)
            train_loader.sampler.update_sampler(hard_pairs)

        else:
            train_loader.sampler.random()

        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)
            plot_frequencies(counters_actions, epoch, exp_path, config)

if __name__ == "__main__":
    main()