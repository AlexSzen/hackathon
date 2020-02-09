""" basic training script """

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from loader import cdl_dataset
import matplotlib.pyplot as plt
from tmp_model import cnn
import os
import copy

def unscale(img):
    """ unscale and return image"""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().detach().numpy()
    return npimg

### Helper to create net directories in results/
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


### helper training function.
def train_model(model, dataloaders, device, criterion, optimizer, num_epochs, result_dir):

    best_weights = copy.deepcopy(model.state_dict())
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):

        print('Epoch %f / %f' %(epoch, num_epochs))

        for phase in ['train', 'val']:

            ### set the model in training or evaluation state
            if phase =='train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_count = 0


            phase_losses = []

            for batchI,(inputs, labels) in enumerate(dataloaders[phase]):

                if device == "cuda":
                    inputs = inputs.cuda()
                    labels = labels.cuda()


                    ### zero the param gradients
                optimizer.zero_grad()

                ### forward, track only for train phase
                with torch.set_grad_enabled(phase=='train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                ### backward + optimize if in training phase
                if phase=='train':
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
                    optimizer.step()

                pastLoss = running_loss*1.0 / (running_count+1e-8)

                running_loss += loss.item() * inputs.size(0)
                running_count += inputs.size(0)


                phase_losses.append(loss.item())

                newLoss = (running_loss*1.0) / (running_count+1e-8)

                if batchI % 20 == 0:

                    print("%06d/%06d" % (batchI+1,len(dataloaders[phase])),end='')
                    print(" Loss %8.5f " % (sum(phase_losses)/len(phase_losses)))



            epoch_loss = running_loss/dataset_sizes[phase]
            print('%s loss = %f' %(phase, epoch_loss))
            if phase == 'train':
                train_loss_list.append(epoch_loss)
                if epoch % 100 == 0:
                    npimg = unscale(inputs)

                    np.save(os.path.join(result_dir,"inputs_train_%03d")%(epoch+1), npimg)
                    np.save(os.path.join(result_dir,"outputs_train_%03d")%(epoch+1), outputs.cpu().detach().numpy())
                    np.save(os.path.join(result_dir,"labels_train_%03d")%(epoch+1), labels.cpu().detach().numpy())
            else :
                val_loss_list.append(epoch_loss)

                if epoch % 100 == 0:
                    npimg = unscale(inputs)

                    np.save(os.path.join(result_dir,"inputs_val_%03d")%(epoch+1), npimg)
                    np.save(os.path.join(result_dir,"outputs_val_%03d")%(epoch+1), outputs.cpu().detach().numpy())
                    np.save(os.path.join(result_dir,"labels_val_%03d")%(epoch+1), labels.cpu().detach().numpy())

            ### take first loss as reference at first epoch
            if phase == 'val' and epoch == 0:
                worst_loss = epoch_loss

            ### keep track of best weights if model improves on validation set
            if phase == 'val' and epoch_loss < worst_loss:
                best_weights = copy.deepcopy(model.state_dict())
                worst_loss = epoch_loss



    ### load best weights into model
    model.load_state_dict(best_weights)
    return model, np.asarray(train_loss_list), np.asarray(val_loss_list)



if __name__ == "__main__":

    # path of data
    data_path = "/data/bmoseley/DPhil/temp/hackathon/data/CMP_facade_DB_base/base/"

    # create result dir
    createFolder("results/")
    result_dir = "results/"

    # cpu or gpu
    device = "cuda"

    # number of epochs
    n_epochs = 3000

    # batch_size
    batch_size = 8

    # Dataset & Dataloader for train and validation
    cdl_datasets = {x: cdl_dataset(data_path, train_split = 0.9 ,split=x ) for x in ['train', 'val']}

    cdl_dataloaders = {x: torch.utils.data.DataLoader(cdl_datasets[x], batch_size =  batch_size, shuffle = True, num_workers = 1) for x in ['train', 'val']}

    dataset_sizes = {x: len(cdl_datasets[x]) for x in ['train', 'val']}

    # Define net
    net = cnn().double()
    # switch to gpu if there is one
    if device == "cuda":
        net = net.cuda()

    # optimizer Adam with default params
    optimizer = optim.Adam(net.parameters())

    # loss: Cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    model, train_loss_list, val_loss_list = train_model(net, cdl_dataloaders, device, criterion, optimizer, n_epochs, result_dir)

    ### save net, losses and info in corresponding net directory
    torch.save(net.cpu(), os.path.join(result_dir,"model.pt"))

    np.save(os.path.join(result_dir,"train_loss.npy"), train_loss_list)
    np.save(os.path.join(result_dir,"val_loss.npy"), val_loss_list)


    F = open(os.path.join(result_dir,"log.txt"), 'w')

    F.write('Info and hyperparameters for run' + '\n')
    F.write('Loss : ' + str(criterion) + '\n')
    F.write('Optimizer : ' + str(optimizer) + '\n')
    F.write('Number of epochs : ' + str(n_epochs) + '\n')
    F.write('Batch size : ' +str(batch_size) + '\n')
    F.write('Net architecture : \n' + str(net))
    F.close()
