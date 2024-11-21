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
import time
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



#train_set_root = os.path.join(os.path.abspath('.'), 'dataset/fake-circuit-data_20230623/fake-circuit-data')
valid_set_root = os.path.join(os.path.abspath('.'), 'code' ,'dataset', 'hidden-real-circuit-data')


device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
model_settings = {'channel': 16, 'en_node_num_list': en_node_num_list, 'de_node_num_list': de_node_num_list,
                    'sample_num': sample_num, 'en_func_type': func_type, 'de_func_type': func_type}
model = Net(gene=best_ind[0],model_settings=model_settings)

model.load_state_dict(torch.load('./model/best_mae_model.pth'))



model.to(device)

valid_set, _ = get_datasets('ICCAD', valid_set_root, False)

valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)

print(len(valid_loader))

model.eval()

mae_score = AverageMeter()
time_score = AverageMeter()

cnt = 0


for _, data in enumerate(valid_loader):
    images, targets = data
    images = images.to(device)
    targets = targets.to(device)
    start_time = time.time()

    preds = model(images)

    inference_time = time.time()-start_time
    time_score.update(inference_time)
    print('time: {}'.format(inference_time))


    ## compute mae ##
    pred_n = preds.detach().cpu().numpy()
    targets_n = targets.detach().cpu().numpy()

    #print(pred_n[0,0,:,:].shape)


    mae = np.mean(abs(pred_n - targets_n))
    ####

    np.savetxt(f'./result/pred_{cnt}.csv',pred_n[0,0,:,:],delimiter=',')
    np.savetxt(f'./result/target_{cnt}.csv',targets_n[0,0,:,:],delimiter=',')


    cnt += 1



    print('mae: {}'.format(mae))

    mae_score.update(mae)


print('Average_mae: {}'.format(mae_score.val))
print('Average_time: {}'.format(time_score.val))