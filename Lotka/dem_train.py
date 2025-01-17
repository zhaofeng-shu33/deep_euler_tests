# -*- coding: utf-8 -*-
# python dem_train.py --data lotka_range_data_2000_100.hdf5 --epoch 50 --print_losses 1
import argparse
import os
import h5py
from datetime import datetime
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.jit
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import MLPs
from utils.plot_utils import plot_loghist
#from utils.scalers import writeStandardScaler
#from utils.scalers import writeMinMaxScaler

torch.set_default_dtype(torch.float64)

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser = argparse.ArgumentParser()
parser.add_argument('--generalized_training', default=False, const=True, nargs='?')
parser.add_argument('--model_type', default='simple_mlp', choices=['simple_mlp', 'embedded'], type=str, help="type of the model to use")
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch', default='1000', type=int, help="Batch size. 0 means training set length.")
parser.add_argument('--epoch', default='1',type=int, help="Number of epochs to train.")
parser.add_argument('--load_model', default='', type=str, help="Path to model dict file to load.")
parser.add_argument('--name', default='',type=str, help="Optional name of the model.")
parser.add_argument('--start_epoch', default='0', type=int, help="Epochs of training of the loaded model. Deprecated")
parser.add_argument('--save_path', default='training/', type=str, help="Path to save model.")
parser.add_argument('--monitor', default=0, type=int, help="0: no monitoring, 1: show plots on end, 2: monitor all along")
parser.add_argument('--print_losses',default=1, type=int, help="Print every nth losses. 0 means no print. Default=1 means prints every epoch. Option monitor=2 overrides this.")
parser.add_argument('--save_plots', dest='save_plots', action='store_true', help="If set, saves the plots generated after training.")
parser.add_argument('--test', dest='test', action='store_true',help="If set, no saving takes place.")
parser.add_argument('--print_epoch', default=0, type=int, help="Print epoch number at every nth epoch. Default is zero, meaning no print.")
parser.add_argument('--cpu', dest='cpu', action='store_true', help= "If set, training is carried out on the cpu.")
parser.add_argument('--early_stop', dest='early_stop', action='store_true', help= "Enable early stop when the latest validation loss is larger than the average of the previous five validation losses.")
parser.add_argument('--num_threads', default=0, type=int, help="Number of cpu threads to be used by pytorch. Default is 0 meaning same as number of cores.")
parser.set_defaults(feature=False, monitor=False, load_model=False, test=False, cpu=False, print_epoch=False, early_stop=False)

args = parser.parse_args()

if args.num_threads:
    torch.set_num_threads(args.num_threads)


# device selection logic
device = 0
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


begin_time = datetime.now()
time_str = begin_time.strftime("%Y_%m_%d") # year month day
print("Begin: " + str(time_str))


#check model availability
if args.load_model:
    if not os.path.exists(args.load_model):
        print("File: " + args.load_model + " does not exist. Abort")
        exit()

# ----- ----- ----- ----- ----- -----
# Data loading
# ----- ----- ----- ----- ----- -----

if args.generalized_training:
    data_path = 'lotka_range_data.hdf5'
else:
    data_path = 'lotka_data2.hdf5'

f = h5py.File(data_path, 'r')
keys = list(f.keys())
print(keys)
X = np.empty(f['lotka_X'].shape)
f['lotka_X'].read_direct(X)
Y = np.empty(f['lotka_Y'].shape)
f['lotka_Y'].read_direct(Y)   
f.close()


print(X[1,:])
input_names = ["x1", "x2"]

if not args.test:                                                                                                                                                                                                             
    print("Train data from: " + data_path)

input_length = X.shape[1]
print("Input length: " +str(input_length))

x_trn, x_vld, y_trn, y_vld  = train_test_split(
    X, Y,
    test_size   = .25,
    random_state= 410,
    shuffle     = True
    )
x_vld, x_tst, y_vld, y_tst  = train_test_split(
    x_vld, y_vld,
    test_size   = .40,
    shuffle     = False
    )



trn_set = TensorDataset(torch.tensor(x_trn, dtype=torch.float64), torch.tensor(y_trn, dtype=torch.float64))
vld_set = TensorDataset(torch.tensor(x_vld, dtype=torch.float64), torch.tensor(y_vld, dtype=torch.float64))
trn_ldr = DataLoader(
    trn_set,
    batch_size  = len(trn_set) if args.batch==0 else args.batch,
    shuffle     = True
    )
vld_batch = args.batch
vld_ldr = DataLoader(
    vld_set,
    batch_size  = vld_batch,
    shuffle     = False
    )

start_epoch = 0
# ----- ----- ----- ----- ----- -----
# Model definition
# ----- ----- ----- ----- ----- -----
if args.model_type == 'embedded' and args.generalized_training is True:
    model = MLPs.SimpleMLPGen(4, y_trn.shape[-1], 80, x_trn.shape[-1] - 4)
else:
    model = MLPs.SimpleMLP(x_trn.shape[-1], y_trn.shape[-1], 80)

model_checkpoint = 0
if args.load_model:
    model_checkpoint = torch.load(args.load_model)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    start_epoch = model_checkpoint['epoch']
    if not args.test:
        print("Loaded model state from: " + str(args.load_model))
        
# ----- ----- ----- ----- ----- -----
# Training
# ----- ----- ----- ----- ----- -----
model = model.to(device)
loss    = nn.MSELoss()
optim   = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)#, weight_decay=1e-7)
if args.load_model:
    optim.load_state_dict(model_checkpoint['optimizer_state_dict'])

total_loss_arr  = np.zeros(args.epoch)
vld_loss_arr    = np.zeros(args.epoch)
epochs          = np.linspace(start_epoch, start_epoch + args.epoch - 1, args.epoch)

if args.monitor==2: 
    plt.ion()
    plt.figure(num="Training and Validation Losses")
    
if not args.test:
    print("Training...")

learned_epoch = 0
vld_loss_best = 1e100
best_model_state_dict = 0
best_optim_state_dict = 0
best_epoch = 0
for num_epoch in range(args.epoch):
    if args.print_epoch and num_epoch % args.print_epoch == 0:
        print('epoch', num_epoch + start_epoch)
    model.train()
    total_loss  = 0
    for batch in trn_ldr:
        x,y = batch
        x   = x.to(device)
        y   = y.to(device)
        optim.zero_grad()
        out     = model(x)
        trn_loss= loss(out, y)
        total_loss  += trn_loss.item() * len(x)
        trn_loss.backward()
        optim.step()
        
    total_loss  /= len(trn_ldr.dataset)
    total_loss_arr[num_epoch] = total_loss
    learned_epoch += 1
    model.eval()
    vld_loss    = 0
    for batch in vld_ldr:
        x, y= batch
        x   = x.to(device)
        y   = y.to(device)
        out     = model(x)
        vld_loss += loss(out, y).item() * len(x)
    vld_loss    /= len(vld_ldr.dataset)
    vld_loss_arr[num_epoch] = vld_loss
    if args.monitor == 2 or (args.print_losses and num_epoch % args.print_losses == 0):
        print(f'{num_epoch}: total loss', total_loss, 'validation loss', vld_loss)

    
    if args.monitor==2: #real-time plotting
        plt.cla()
        plt.plot(epochs, total_loss_arr)
        plt.plot(epochs, vld_loss_arr)
        plt.yscale('log')
        if num_epoch!= 0: plt.xlim([start_epoch, start_epoch+num_epoch])
        plt.pause(0.01)

    if args.early_stop:
        if vld_loss < vld_loss_best:
            vld_loss_best = vld_loss
            best_model_state_dict = deepcopy(model.state_dict())
            best_optim_state_dict = deepcopy(optim.state_dict())
            best_epoch = num_epoch
        else:
            if num_epoch-best_epoch == 50:
                if not args.test:
                    print("Early stopped")
                print("Early stopped")
                break
    
if not args.test:
    print("Training ready, epochs: " + str(start_epoch) + "..." + str(start_epoch + learned_epoch))

end_time = datetime.now()
duration = end_time - begin_time
time_end_str = end_time.strftime("%y%m%d%H%M")
print("Ended at: "+ time_end_str)
print("Duration: " + str(duration))
if not args.test:
    print("Training duration: " + str(duration))

# ----- ----- ----- ----- ----- -----
#Test
# ----- ----- ----- ----- ----- -----
tst_set = TensorDataset(torch.Tensor(x_tst), torch.Tensor(y_tst))
tst_batch = args.batch
tst_ldr = DataLoader(
    tst_set,
    batch_size  = tst_batch,
    shuffle     = False
    ) 

test_loss    = 0
for batch in tst_ldr:
    x, y= batch
    x   = x.to(device)
    y   = y.to(device)
    out = model(x)
    test_loss += loss(out, y).item() * len(x)
test_loss    /= len(tst_ldr.dataset)
if not args.test:
    print('Test L2 loss: ' + str(test_loss))

# fix me: SimpleMLPGen does not support huge batch size
if args.model_type != 'embedded':
    out = model(torch.tensor(x_tst, dtype=torch.float64).to(device)).cpu().detach().numpy()
    test_losses = np.abs(out - y_tst) # L1 loss
    max_loss = np.max(test_losses)
    mean_loss = np.mean(test_losses)
    if not args.test:
        print('Max unnormalized L1 loss: ' + str(max_loss))
        print('Mean unnormalized L1 loss: ' + str(mean_loss))


# ----- ----- ----- ----- ----- -----
# Model Save
# ----- ----- ----- ----- ----- -----

traced_model = 0
if not args.test:
    saved_prefix = 'model_'
    if args.generalized_training:
        saved_prefix = 'range_model_'
        if args.model_type == 'embedded':
            saved_prefix = 'range_embedded_model_'
    if args.early_stop:
        torch.save({
            'epoch': start_epoch+best_epoch,
            'model_state_dict': best_model_state_dict,
            #'scheduler_state_dict': best_scheduler_state_dict,
            'optimizer_state_dict': best_optim_state_dict
            },
            args.save_path + saved_prefix + (args.name+'_' if args.name else '') + 'e' + str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    else:
        torch.save({
            'epoch': start_epoch+learned_epoch,
            'model_state_dict': model.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            },
            args.save_path + saved_prefix + (args.name+'_' if args.name else '') + 'e' + str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    print("Saving model.")

    # trace model to be used by C/C++
    if args.early_stop:
        model.load_state_dict(best_model_state_dict)
        model.eval()
    if args.model_type == 'embedded':
        meta_features = torch.Tensor([1.0, 1, 1, 1, 2.0, 1.0])
        p_model = MLPs.SimpleMLPGen_with_meta_feature(4, 2, 80)
        p_model.set_parameters(meta_features, model)
        traced_model = torch.jit.trace(p_model.cpu(), torch.randn((1, 4)))
    else:
        traced_model = torch.jit.trace(model.cpu(), torch.randn((1,x_trn.shape[-1])))
    traced_model.save(args.save_path + 'traced_' + saved_prefix + (args.name+'_' if args.name else '') + 'e'+str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    print("Saving trace model.")

# ----- ----- ----- ----- ----- -----
# Plotting
# ----- ----- ----- ----- ----- -----
 
if args.monitor > 0:
    plt.ion()
    plt.show()
plt.plot(epochs[0:learned_epoch], total_loss_arr[0:learned_epoch], label='Total Loss')
plt.plot(epochs[0:learned_epoch], vld_loss_arr[0:learned_epoch], label='Validation Loss')
plt.yscale('log')
plt.title('Loss Diagram')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
if args.monitor > 0:
    plt.show()
if not args.test and args.save_plots:
    plt.savefig(args.save_path+"learning_curve_"+ (args.name+'_' if args.name else '') + time_str+".png", transparent=True)


if args.model_type != 'embedded':
    plt.figure(num="Losses")
    plt.title("Loss Distribution of Truncation Error")
    for i in range(test_losses.shape[1]):
        plot_loghist(test_losses[:,i], 500, label=input_names[i])
    plt.legend()
    if args.monitor > 0:
            plt.show()
    if not args.test and args.save_plots:
        plt.savefig(args.save_path+"Loss_distr_"+time_str+".png", transparent=True)
            
    plt.figure(num="Losses (Full)")
    plt.title("Loss Distribution of Truncation Error(Full)")
    plot_loghist(test_losses.flat, 500)
    #plt.hist(test_losses.flat, bins=50)
    #plt.ylim([0,500])
    plt.xscale('log')
    if args.monitor>0:
        plt.ioff()
        plt.show()
    if not args.test and args.save_plots:
        plt.savefig(args.save_path+"Loss_distr_full_"+time_str+".png", transparent=True)
