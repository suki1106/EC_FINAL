import numpy as np
from torchvision import transforms
from model.genetic_unet.genetic_unet import Net
import torch
import pickle
import os
from metrics.average_meter import AverageMeter

from deap import tools
from deap import base
from deap import creator
from train.util.get_optimizer import get_optimizer
from dataset.util.get_datasets import get_datasets
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_




import torch.nn as nn
pickle_file = open(os.path.join(os.path.abspath('.'), 'exps/test/pickle/gens5_ckpt.pkl'), 'rb')
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## To minimize mse
# creator.create("Individual", list, fitness=creator.FitnessMin)
    
genes = pickle.load(pickle_file)
invalid_ind = [ind for ind in genes if not ind.fitness.valid]
best_ind = tools.selBest(genes, 1)
# print(best_idx)
# print(best_idx[0].fitness.values[0])

#{'gens_0_individual_0': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]}
#{'gens_0_individual_1': [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1]}
sample_num = 3
en_node_num = 5
de_node_num = 5
en_node_num_list = [en_node_num for _ in range(sample_num + 1)]
de_node_num_list = [de_node_num for _ in range((sample_num))]

func_type = ['conv_relu_3', 'conv_mish_3', 'conv_in_relu_3',
                'conv_in_mish_3', 'p_conv_relu_3', 'p_conv_mish_3',
                'p_conv_in_relu_3', 'p_conv_in_mish_3', 'conv_relu_5',
                'conv_mish_5', 'conv_in_relu_5','conv_in_mish_5', 'p_conv_relu_5',
                'p_conv_mish_5','p_conv_in_relu_5', 'p_conv_in_mish_5']
optimizer_name = 'Lookahead(Adam)'
learning_rate = 0.001
l2_weight_decay = 0

epochs = 200

train_set_root = os.path.join(os.path.abspath('.'), 'dataset/fake-circuit-data_20230623/fake-circuit-data')
valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', 'real-circuit-data_20230615')


device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
model_settings = {'channel': 64, 'en_node_num_list': en_node_num_list, 'de_node_num_list': de_node_num_list,
                    'sample_num': sample_num, 'en_func_type': func_type, 'de_func_type': func_type}
model = Net(gene=best_ind[0],model_settings=model_settings)

model.to(device)

loss_func = nn.L1Loss().to(device) ## mae loss
optimizer = get_optimizer(optimizer_name, filter(lambda p: p.requires_grad, model.parameters()), learning_rate, l2_weight_decay)

train_set, num_return = get_datasets('ICCAD', train_set_root, True)
valid_set, _ = get_datasets('ICCAD', valid_set_root, False)

train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=3)
valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)


best_mae = 999999999.0

for i in range(epochs):
    model.train()

    for _, data in enumerate(train_loader):

        inputs, targets = data

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = loss_func(preds,targets)

        loss.backward()

        clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()


    print('epoch {} train end'.format(i))

    epoch_mae = AverageMeter()

    with torch.no_grad():
        model.eval()
        for _, data in enumerate(valid_loader):
            inputs, targets = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            loss = loss_func(preds,targets)

            epoch_mae.update(loss.item())

        
        print('epoch {} validation end'.format(i))
        print('mae: {}'.format(epoch_mae.val))

        if epoch_mae.val < best_mae:
            best_mae = epoch_mae.val

            # save model

            torch.save(model.state_dict(), './model/best_mae_model_32.pth')


        print('current_best_mae: {}'.format(best_mae))
