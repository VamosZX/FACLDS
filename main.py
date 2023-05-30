# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.data_utils import DatasetFAC, create_dataset_and_evalmetrix
from utils.util import Partial_Client_Selection, valid, ewc_prepare, ewc_average_model, emotion_prior_loss
from utils.start_config import initization_configure

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    # Configuration for FedAVG, prepare model, optimizer, scheduler
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    model_avg = deepcopy(model).cpu()

    print("=============== Running training ===============")

    loss_fct = torch.nn.CrossEntropyLoss()
    tot_clients = args.dis_cvs_files
    epoch = -1
    while True:
    
        epoch += 1
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0 
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        test_loader_proxy_clients = {}

        importance_matrices = {} 

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_single_client

            trainset = DatasetFAC(args, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)
            
            m = model_all[proxy_single_client]
            m = m.to(args.device).train()
            optimizer = optimizer_all[proxy_single_client]
            scheduler = scheduler_all[proxy_single_client]
            if args.decay_type == 'step':
                scheduler.step()

            print('Train the client', cur_single_client, 'of communication round', epoch)

            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader): 
                    args.global_step_per_client[proxy_single_client] += 1
                    batch = tuple(t.to(args.device) for t in batch)
                    optimizer.zero_grad()
                    x, y = batch
                    predict = m(x)
                    loss_ep = emotion_prior_loss(args,predict,y)
                    loss_ce = torch.sum(loss_fct(predict, y)) / y.size(0)
                    loss = (1-args.lambda)*loss_ce + args.lambda*loss_ep
                    loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if not args.decay_type == 'step':
                        scheduler.step()
                    optimizer.step()

                    writer.add_scalar(proxy_single_client + '/lr', scalar_value=optimizer.param_groups[0]['lr'],
                                      global_step=args.global_step_per_client[proxy_single_client])
                    writer.add_scalar(proxy_single_client + '/loss', scalar_value=loss.item(),
                                      global_step=args.global_step_per_client[proxy_single_client])
                    args.learning_rate_record[proxy_single_client].append(optimizer.param_groups[0]['lr'])
                    if (step+1 ) % 10 == 0:
                        print(cur_single_client, step,':', len(train_loader),'inner epoch', inner_epoch, 'round', epoch,':',
                              args.max_communication_rounds, 'loss', loss.item(), 'lr', optimizer.param_groups[0]['lr'])

            _,_,importance_matrices[args.single_client] = ewc_prepare(args,m)

            print("Update the importance_matrices of client ",cur_single_client)
            m.eval()
            for step, batch in enumerate(train_loader):
                m.zero_grad()
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                predict = m(x)
                loss = loss_fct(predict[0],y[0])   
                loss.backward()  
                for n, p in m.named_parameters():  
                    if n in importance_matrices[args.single_client].keys():
                        importance_matrices[args.single_client][n].data += p.grad.data ** 2 / len(train_loader)
            m.train()

        ## model average

        ewc_average_model(args,model_avg,model_all,importance_matrices)

        if epoch % 10 == 0:
            testset = DatasetFAC(args, phase = 'test')
            test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=0)

        # then evaluate
            for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
                args.single_client = cur_single_client
                model = deepcopy(model_all[proxy_single_client])
                model.to(args.device)
                valid(args, model, test_loader)
                model.cpu()
        

        args.record_val_acc = args.record_val_acc.append(args.current_acc, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)

        tmp_round_acc = [val for val in args.current_test_acc.values() if not val == []]
        writer.add_scalar("test/average_accuracy", scalar_value=np.asarray(tmp_round_acc).mean(), global_step=epoch)

        if epoch == args.communication_rounds:
            break

    writer.close()
    print("================End training================ ")


def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="MLP",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--FL_platform", type = str, default="MLP-FACLDS", choices=["MLP-FACLDS"],  help="Choose of different FL platform. ")
    parser.add_argument("--dataset", choices=["afew","mead","youtube"], default="youtube", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='/data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=True,  help="Save the best model for each client.")
    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="/output_dir/..", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="/output_dir/", type=str, help="The output directory where checkpoints/results/logs will be written.")
    
    parser.add_argument("--optimizer_type", default="adamw",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=0, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=128, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0  0,1,2")
    parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization") 
    parser.add_argument("--lambda", default=0.3, type=float,  help="Hyperparameter")

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=0.00001, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    
    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=300, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    
    parser.add_argument("--split_type", type=str, choices=["c_5_a_1", "c_10_a_1", "c_5_a_5","c_10_a_5"], default="c_5_a_1", help="Which data partitions to use")


    args = parser.parse_args()

    # Initialization

    model = initization_configure(args)

    train(args, model)


    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union test accuracy is: %2.5f  \n' %  \
                   (np.asarray(list(args.best_acc.values())).mean())
    message += "================ End ================ \n"


    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)




if __name__ == "__main__":
    main()
