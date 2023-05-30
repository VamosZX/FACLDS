from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from utils.scheduler import setup_scheduler


## for optimizaer

from torch import optim as optim


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_%s_checkpoint.bin" % (args.FL_platform, client_name,args.split_type))

    torch.save(model.state_dict(),'/output_dir/model.pt')

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def inner_valid(args, model, test_loader):
    eval_losses = AverageMeter()
    print("++++++ Running Test of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []
    
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            if args.num_classes > 1:
                eval_loss = loss_fct(logits.view(-1, args.num_classes), y.view(-1))
                ep_loss = torch.squeeze(emotion_prior_loss(args,logits,y))

                eval_loss = (1-args.lambda)*eval_loss+args.lambda*ep_loss
                eval_losses.update(eval_loss.item())

            if args.num_classes > 1:
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = logits

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
    all_preds, all_label = all_preds[0], all_label[0]

    if not args.num_classes == 1:
        eval_result = simple_accuracy(all_preds, all_label)
    else:
        eval_result = mean_squared_error(all_preds, all_label)

    model.train()

    return eval_result, eval_losses

def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag

def valid(args, model, test_loader):
    # Validation
    eval_result, eval_losses = inner_valid(args, model, test_loader)
    print("Test Loss: %2.5f" % eval_losses.avg, "Test metric: %2.5f" % eval_result)
    
    if metric_evaluation(args, eval_result):
        if args.save_model_flag:
            save_model(args, model)
        
        args.best_acc[args.single_client] = eval_result
        args.best_eval_loss[args.single_client] = eval_losses.val
        print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])

    else:
        print("Donot replace previous best metric of client", args.best_acc[args.single_client])

    args.current_acc[args.single_client] = eval_result

def optimization_func(args, model):

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")
    return optimizer


def Partial_Client_Selection(args, model):

    # Select partial clients join in FL train
    if args.num_local_clients == -1: 
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients =  len(args.dis_cvs_files)
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    scheduler_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    for proxy_single_client in args.proxy_clients:
        model_all[proxy_single_client] = deepcopy(model).cpu()
        optimizer_all[proxy_single_client] = optimization_func(args, model_all[proxy_single_client])
        args.t_total[proxy_single_client] = args.clients_with_len[proxy_single_client] *  args.max_communication_rounds / args.batch_size * args.E_epoch
        scheduler_all[proxy_single_client] = setup_scheduler(args, optimizer_all[proxy_single_client], t_total=args.t_total[proxy_single_client])
        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    return model_all, optimizer_all, scheduler_all


def ewc_prepare(args, model):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad} 
    _means = {} 
    if "MLP" in args.FL_platform:
        del params['layer_out.weight']
        del params['layer_out.bias']
 
    for n, p in params.items():
        _means[n] = p.clone().detach()
    importance_matrices = {} 
    for n, p in params.items():
        importance_matrices[n] = p.clone().detach().fill_(0) 

    return params,_means,importance_matrices


def ewc_average_model(args,  model_avg, model_all, i_m):
    model_avg.cpu()
    print('Calculate the model ewc_avg----')
    params = dict(model_avg.named_parameters()) 
    if "MLP" in args.FL_platform:
        del params['layer_out.weight']
        del params['layer_out.bias']

    for client in i_m.keys():
        for layer in i_m[client].keys():
            i_m[client][layer] = torch.nn.Softmax(dim=-1)(i_m[client][layer])

    if args.split_type=='c_5_a_1' or args.split_type=='c_5_a_5':
        c_1 = i_m["train_1"]
        c_2 = i_m["train_2"]
        c_3 = i_m["train_3"]
        c_4 = i_m["train_4"]
        c_5 = i_m["train_5"]
        for name in c_1.keys():
            shapes = []
            shapes.append(5)
            for i in c_1[name].size():
                shapes.append(i)
            c_1[name],c_2[name],c_3[name],c_4[name],c_5[name] = \
            torch.softmax(
                torch.stack(
                    (c_1[name].reshape(-1),
                    c_2[name].reshape(-1),
                    c_3[name].reshape(-1),
                    c_4[name].reshape(-1),
                    c_5[name].reshape(-1)),dim=0),dim=0).reshape(shapes).split(1,dim=0)
        i_m["train_1"] = c_1
        i_m["train_2"] = c_2
        i_m["train_3"] = c_3
        i_m["train_4"] = c_4
        i_m["train_5"] = c_5

        for name, param in params.items():
            for client in range(len(args.proxy_clients)):
                single_client = args.proxy_clients[client]
                if client == 0:
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                        name].data.to(args.device) * i_m[single_client][name].to(args.device)
                else:
                    tmp_param_data = tmp_param_data + \
                                    dict(model_all[single_client].named_parameters())[
                                        name].data.to(args.device) * i_m[single_client][name].to(args.device)
            params[name].data.copy_(torch.squeeze(tmp_param_data,dim=0))

        print('Update each client model parameters----')

        for single_client in args.proxy_clients:
            tmp_params = dict(model_all[single_client].named_parameters())
            for name, param in params.items():
                tmp_params[name].data.copy_(param.data)

    elif args.split_type=='c_10_a_1' or args.split_type=='c_10_a_5':
        c_1 = i_m["train_1"]
        c_2 = i_m["train_2"]
        c_3 = i_m["train_3"]
        c_4 = i_m["train_4"]
        c_5 = i_m["train_5"]
        c_6 = i_m["train_6"]
        c_7 = i_m["train_7"]
        c_8 = i_m["train_8"]
        c_9 = i_m["train_9"]
        c_10 = i_m["train_10"]

        for name in c_1.keys():
            shapes = []
            shapes.append(10)
            for i in c_1[name].size():
                shapes.append(i)
            c_1[name],c_2[name],c_3[name],c_4[name],c_5[name] ,c_6[name],c_7[name],c_8[name],c_9[name],c_10[name]= \
            torch.softmax(
                torch.stack(
                    (c_1[name].reshape(-1),
                    c_2[name].reshape(-1),
                    c_3[name].reshape(-1),
                    c_4[name].reshape(-1),
                    c_5[name].reshape(-1),
                    c_6[name].reshape(-1),
                    c_7[name].reshape(-1),
                    c_8[name].reshape(-1),
                    c_9[name].reshape(-1),
                    c_10[name].reshape(-1)),dim=0),dim=0).reshape(shapes).split(1,dim=0)
        i_m["train_1"] = c_1
        i_m["train_2"] = c_2
        i_m["train_3"] = c_3
        i_m["train_4"] = c_4
        i_m["train_5"] = c_5
        i_m["train_6"] = c_6
        i_m["train_7"] = c_7
        i_m["train_8"] = c_8
        i_m["train_9"] = c_9
        i_m["train_10"] = c_10

        for name, param in params.items():
            for client in range(len(args.proxy_clients)):
                single_client = args.proxy_clients[client]

                if client == 0:
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                        name].data.to(args.device) * i_m[single_client][name].to(args.device)
                else:
                    tmp_param_data = tmp_param_data + \
                                    dict(model_all[single_client].named_parameters())[
                                        name].data.to(args.device) * i_m[single_client][name].to(args.device)
            params[name].data.copy_(torch.squeeze(tmp_param_data,dim=0))

        print('Update each client model parameters----')

        for single_client in args.proxy_clients:
            tmp_params = dict(model_all[single_client].named_parameters())
            for name, param in params.items():
                tmp_params[name].data.copy_(param.data)


def emotion_prior_loss(args, predict, target):
    theta, r, phi, p = label_turn_hemiphere(args, target)
    theta0,r0,phi0,p0 = pred_turn_hemiphere(args, predict)
    loss = ep_loss(args, theta, r, phi, p, theta0, r0, phi0, p0)
    return loss
  

def label_turn_hemiphere(args, target): 
    theta = []
    r =[]
    phi = []
    p = []
    if args.dataset == 'afew':
        for i in target:
            if i < args.class_num-1: 
                theta.append(torch.tensor((((i+1)-1)/3)*math.pi).to(args.device))
                r.append(torch.tensor(1).to(args.device))
                phi.append(torch.tensor(7*math.pi/12).to(args.device))
                if i==0 or i==2:
                    p.append(torch.tensor(-1).to(args.device))
                else:
                    p.append(torch.tensor(1).to(args.device))
            else:
                theta.append(torch.tensor(0).to(args.device))
                r.append(torch.tensor(1).to(args.device))
                phi.append(torch.tensor(0).to(args.device))
                p.append(torch.tensor(0).to(args.device))
    elif args.dataset == 'mead':
        for i in target:
            if i < args.class_num-1:
                theta.append(torch.tensor(((2*(i+1))/7)*math.pi).to(args.device))
                r.append(torch.tensor(1).to(args.device))
                phi.append(torch.tensor(7*math.pi/12).to(args.device))
                if (math.pi/7)<theta[-1]<((5*math.pi)/7):
                    p.append(torch.tensor(1).to(args.device))
                else:
                    p.append(torch.tensor(-1).to(args.device))
            else:
                theta.append(torch.tensor(0).to(args.device))
                r.append(torch.tensor(1).to(args.device))
                phi.append(torch.tensor(0).to(args.device))
                p.append(torch.tensor(0).to(args.device))
    elif args.dataset == 'youtube':
        for i in target:  
            theta.append(torch.tensor(((2*(i%8+1)-1)/8)*math.pi).to(args.device))
            r.append(torch.tensor(1).to(args.device))
            if i>=0 and i<8:
                phi.append(torch.tensor(math.pi/12).to(args.device))
            elif i>=8 and i<16:
                phi.append(torch.tensor(3 * math.pi/12).to(args.device))
            else:
                phi.append(torch.tensor(5 * math.pi/12).to(args.device))
            
            if 0<=theta[-1]<((math.pi)/2) or ((3 * math.pi)/2)<=theta[-1]<(2 * math.pi):
                p.append(torch.tensor(1).to(args.device))
            else:
                p.append(torch.tensor(-1).to(args.device))
          
    return theta, r, phi, p


def pred_turn_hemiphere(args, predict):
    theta = []
    r =[]
    phi = []
    p = []
    if args.dataset == 'afew':
        for i in predict:
            x = []
            y = []
            z = []
            for j in range(args.class_num):
                if j < args.class_num-1:
                    theta0 = (((j+1)-1)/3)*math.pi
                    r0 = 1
                    phi0 = 7*math.pi/12
                    
                else:
                    theta0 = 0
                    r0 = 1
                    phi0 = 0
                r1 = r0 * i[j]

                x.append(r1*math.sin(phi0)*math.cos(theta0))
                y.append(r1*math.sin(theta0)*math.sin(phi0))
                z.append(r1*math.cos(phi0))

            sum_x = sum(torch.Tensor(x))
            sum_y = sum(torch.Tensor(y))
            sum_z = sum(torch.Tensor(z))

            r2 = torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2)+torch.pow(sum_z,2))
            theta1 = torch.atan(sum_y/sum_x)
            phi1 = torch.atan(torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2))/sum_z)

            theta.append(theta1)
            r.append(r2)
            phi.append(phi1)
            if torch.pi/2<theta1<(7*torch.pi)/6 and phi1 != torch.tensor(0):
                p.append(torch.tensor(1).to(args.device))
            elif theta1==torch.tensor(0).to(args.device) and phi1==torch.tensor(0).to(args.device):
                p.append(torch.tensor(0).to(args.device))
            else:
                p.append(torch.tensor(-1).to(args.device))
    elif args.dataset == 'mead':
        for i in predict:
            x = []
            y = []
            z = []
            for j in range(args.class_num):
                if j < args.class_num-1:
                    theta0 = ((2*(j+1))/7)*math.pi
                    r0 = 1
                    phi0 = 7*math.pi/12
                else:
                    theta0 = 0
                    r0 = 1
                    phi0 = 0
                r1 = r0 * i[j]
                x.append(r1*math.sin(phi0)*math.cos(theta0))
                y.append(r1*math.sin(theta0)*math.sin(phi0))
                z.append(r1*math.cos(phi0))

            sum_x = sum(torch.Tensor(x))
            sum_y = sum(torch.Tensor(y))
            sum_z = sum(torch.Tensor(z))

            r2 = torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2)+torch.pow(sum_z,2))
            theta1 = torch.atan(sum_y/sum_x)
            phi1 = torch.atan(torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2))/sum_z)

            theta.append(theta1)
            r.append(r2)
            phi.append(phi1)
            if  (torch.pi/7)<theta1<((5*torch.pi)/7) and phi1!= torch.tensor(0):
                p.append(torch.tensor(1).to(args.device))
            elif theta1==torch.tensor(0).to(args.device) and phi1==torch.tensor(0).to(args.device):
                p.append(torch.tensor(0).to(args.device))
            else:
                p.append(torch.tensor(-1).to(args.device))
    elif args.dataset == 'youtube':
        for i in predict:
            x = []
            y = []
            z = []
            for j in range(args.class_num):
               
                theta0 = ((2*(j%8+1)-1)/8)*math.pi
                r0 = 1
                if j>=0 and j<8:
                    phi0 = math.pi/12
                elif j>=8 and j<16:
                    phi0 = 3 * math.pi/12
                else:
                    phi0 = 5 * math.pi/12 
                r1 = r0 * i[j]
                x.append(r1*math.sin(phi0)*math.cos(theta0))
                y.append(r1*math.sin(theta0)*math.sin(phi0))
                z.append(r1*math.cos(phi0))

            sum_x = sum(torch.Tensor(x))
            sum_y = sum(torch.Tensor(y))
            sum_z = sum(torch.Tensor(z))

            r2 = torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2)+torch.pow(sum_z,2))
            theta1 = torch.atan(sum_y/sum_x)
            phi1 = torch.atan(torch.sqrt(torch.pow(sum_x,2)+torch.pow(sum_y,2))/sum_z)

            theta.append(theta1)
            r.append(r2)
            phi.append(phi1)
            if  0<=theta1<((torch.pi)/2) or ((3 * torch.pi)/2)<=theta1<(2 * torch.pi):    
                p.append(torch.tensor(1).to(args.device))
            else:
                p.append(torch.tensor(-1).to(args.device))

    return theta,r,phi,p


def ep_loss(args, theta, r, phi, p, theta0, r0, phi0, p0):
    loss = 0
    for  a,b,c,d,e,f,g in zip(theta,theta0,phi,phi0,p,p0,r0):
        
        loss += g*(torch.pow((a-b),2)+torch.pow((c-d),2)+torch.pow((e-f),2))
    loss = loss/len(theta)
    return loss
    


    

         



