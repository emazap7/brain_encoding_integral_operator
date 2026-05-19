#General libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import scipy
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision
from nilearn import image
from nilearn import plotting
from nilearn.image import threshold_img

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

#Custom libraries
from IE_source.utils import EarlyStopping, SaveBestModel, to_np, LRScheduler, load_checkpoint, relative_mse_loss, log_cosh_loss, VGGPerceptualLoss, variance_regularization_loss, compute_r2, compute_pearson
from torch.utils.data import SubsetRandomSampler
from IE_source.Attentional_IE_solver import Integral_spatial_attention_solver_multbatch
from IE_source.utils import plot_reconstruction

#Torch libraries
import torch
from torch.nn import functional as F

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"






def Brain_imaging_classification(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)
        
    spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.shapes[i+1]) for i in range(2)])
    x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
    y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
    
    spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()

        loss_func = torch.nn.CrossEntropyLoss(weight=args.class_weights)
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for obs_, labels_ in tqdm(train_loader):
                
                obs_ = obs_.to(args.device)
                labels_ = labels_.to(args.device)

                c= lambda x: Encoder(obs_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(obs_)[...,-1:,:].to(args.device)
                
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_[...,-1:,:].requires_grad_(True))
                    
                loss = loss_func(z_, labels_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, labels_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for obs_val, labels_val in tqdm(valid_loader):
                    
                    obs_val = obs_val.to(args.device)
                    labels_val = labels_val.to(args.device)
    
                    c= lambda x: Encoder(obs_val).to(args.device)
                    y_0 =  Encoder(obs_val)[...,-1:,:].to(args.device)
                    
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= spatial_domain.to(args.device),
                                        spatial_domain_dim=2,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val[...,-1:,:])
                         
                    loss_validation = loss_func(z_val, labels_val)
                        
                    del labels_val, z_val, obs_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            for obs_test, labels_test in tqdm(test_loader):
                
                obs_test = obs_test.to(args.device)
                labels_test = labels_test.to(args.device)

                c= lambda x: Encoder(obs_test).to(args.device)
                y_0 =  Encoder(obs_test)[...,-1:,:].to(args.device)
                
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    ).solve()

                
                z_test = Decoder(z_test[...,-1:,:])
                
                _, predicted = torch.max(z_test.data, -1)
                n_samples += labels_test.size(0)
                n_correct += (predicted == labels_test).sum().item() 

                error += torch.abs(predicted-labels_test).sum().item()
                
                tot_predicted = torch.cat([tot_predicted,predicted], dim=0)
                tot_labels = torch.cat([tot_labels, labels_test], dim=0)
                

                del z_test, labels_test, obs_test

            avg_error = error/n_samples
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy: {acc} %')
            print(f'Average Error: {avg_error}')
            print(classification_report(to_np(tot_labels), to_np(tot_predicted)))
            print(tot_predicted)
            print(torch.abs(tot_predicted-tot_labels))
            # print(roc_curve(to_np(tot_labels),to_np(tot_predicted)))



def Brain_self_supervised(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)
        
    spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.shapes[i+1]) for i in range(2)])
    x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
    y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
    
    spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()

        loss_func = F.mse_loss
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for obs_, labels_ in tqdm(train_loader):

                obs_ = obs_.to(device)
                obs_in = obs_ + args.std*torch.rand_like(obs_).to(device)
                obs_in = obs_in/torch.abs(obs_in).max()
                labels_ = labels_.to(args.device)


                c= lambda x: Encoder(obs_in.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(obs_in)[...,-1:,:].to(args.device)
                
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    embedding=args.embedding
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_[...,1:,:].requires_grad_(True)).view_as(obs_)
                    
                loss = loss_func(z_, obs_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, labels_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for obs_val, labels_val in tqdm(valid_loader):
                    
                    obs_val = obs_val.to(device)
                    obs_val_in = obs_val + args.std*torch.rand_like(obs_val).to(device)
                    obs_val_in = obs_val_in/torch.abs(obs_val_in).max()
                    labels_val = labels_val.to(args.device)
    
                    c= lambda x: Encoder(obs_val_in).to(args.device)
                    y_0 =  Encoder(obs_val_in)[...,-1:,:].to(args.device)
                    
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= spatial_domain.to(args.device),
                                        spatial_domain_dim=2,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        embedding=args.embedding
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val[...,1:,:]).view_as(obs_val)
                         
                    loss_validation = loss_func(z_val, obs_val)

                    if args.plot_pred==True and counter==0:
                        vid_plt = z_val[0,...,:1].cpu()
                        obs_plot = obs_val[0,...,:1].cpu()
                        vid_plt = vid_plt.permute(2,3,0,1)
                        obs_plot = obs_plot.permute(2,3,0,1)
                        grid_img0 = torchvision.utils.make_grid(vid_plt, nrow=4)
                        grid_img1 = torchvision.utils.make_grid(obs_plot, nrow=4)
                        plt.figure(figsize=(10,10))
                        plt.imshow(grid_img0.permute(1, 2, 0))
                        plt.show()
                        plt.figure(figsize=(10,10))
                        plt.imshow(grid_img1.permute(1, 2, 0))
                        plt.show()
                        
                    del labels_val, z_val, obs_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.

        z_embedded = torch.tensor([]).to(device)
        with torch.no_grad():

            #train to use dropout during inference
            model.train()
            Encoder.train()
            Decoder.train()
                
            n_correct = 0
            n_samples = 0

            for obs_test, labels_test in tqdm(test_loader):
                
                obs_test = obs_test.to(device)
                obs_test_in = obs_test + args.std*torch.rand_like(obs_test).to(device)
                obs_test_in = obs_test_in/torch.abs(obs_test_in).max()
                labels_test = labels_test.to(args.device)

                c= lambda x: Encoder(obs_test_in).to(args.device)
                y_0 =  Encoder(obs_test_in)[...,-1:,:].to(args.device)
                
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=False,
                                    embedding=args.embedding
                                    ).solve()

                z_embedded = torch.cat([z_embedded,z_test[...,0,:]],dim=0)

            return z_embedded
                
            #     _, predicted = torch.max(z_test.data, -1)
            #     n_samples += labels_test.size(0)
            #     n_correct += (predicted == labels_test).sum().item() 

            #     error += torch.abs(predicted-labels_test).sum().item()
                
            #     tot_predicted = torch.cat([tot_predicted,predicted], dim=0)
            #     tot_labels = torch.cat([tot_labels, labels_test], dim=0)
                

            #     del z_test, labels_test, obs_test

            # avg_error = error/n_samples
            # acc = 100.0 * n_correct / n_samples
            # print(f'Accuracy: {acc} %')
            # print(f'Average Error: {avg_error}')
            # print(classification_report(to_np(tot_labels), to_np(tot_predicted)))
            # print(tot_predicted)


def Brain_imaging_classification_3D(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)
        
    spatial_domain_xyz = torch.meshgrid([torch.linspace(0,1,args.shapes[i+1]) for i in range(3)])
    x_space = spatial_domain_xyz[0].flatten().unsqueeze(-1)
    y_space = spatial_domain_xyz[1].flatten().unsqueeze(-1)
    z_space = spatial_domain_xyz[2].flatten().unsqueeze(-1)

    spatial_domain = torch.cat([x_space,y_space,z_space],-1)
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()

        loss_func = torch.nn.CrossEntropyLoss(weight=args.class_weights)
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for obs_, labels_ in tqdm(train_loader):
                
                obs_ = obs_.to(args.device)
                labels_ = labels_.to(args.device)

                c= lambda x: Encoder(obs_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(obs_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = y_0.repeat([1,1,1,1,args.time_points,1])
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=3,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_[...,-1:,:].requires_grad_(True))
                    
                loss = loss_func(z_, labels_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, labels_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for obs_val, labels_val in tqdm(valid_loader):
                    
                    obs_val = obs_val.to(args.device)
                    labels_val = labels_val.to(args.device)
    
                    c= lambda x: Encoder(obs_val).to(args.device)
                    y_0 =  Encoder(obs_val)[...,-1:,:].to(args.device)
                    y_init = y_0.repeat([1,1,1,1,args.time_points,1])
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= spatial_domain.to(args.device),
                                        spatial_domain_dim=3,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val[...,-1:,:])
                         
                    loss_validation = loss_func(z_val, labels_val)
                        
                    del labels_val, z_val, obs_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            for obs_test, labels_test in tqdm(test_loader):
                
                obs_test = obs_test.to(args.device)
                labels_test = labels_test.to(args.device)

                c= lambda x: Encoder(obs_test).to(args.device)
                y_0 =  Encoder(obs_test)[...,-1:,:].to(args.device)
                y_init = y_0.repeat([1,1,1,1,args.time_points,1])
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=3,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                
                z_test = Decoder(z_test[...,-1:,:])
                
                _, predicted = torch.max(z_test.data, -1)
                n_samples += labels_test.size(0)
                n_correct += (predicted == labels_test).sum().item() 

                error += torch.abs(predicted-labels_test).sum().item()
                
                tot_predicted = torch.cat([tot_predicted,predicted], dim=0)
                tot_labels = torch.cat([tot_labels, labels_test], dim=0)
                

                del z_test, labels_test, obs_test

            avg_error = error/n_samples
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy: {acc} %')
            print(f'Average Error: {avg_error}')
            print(classification_report(to_np(tot_labels), to_np(tot_predicted)))
            print(tot_predicted)
            print(torch.abs(tot_predicted-tot_labels))
            # print(roc_curve(to_np(tot_labels),to_np(tot_predicted)))


def Huxby_brain_decoding(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.AdamW(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()

        loss_func = torch.nn.CrossEntropyLoss(args.class_weights)
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for obs_, labels_ in tqdm(train_loader):
                
                obs_ = obs_.to(args.device)
                labels_ = labels_.to(args.device)

                class_indices_ = torch.argmax(labels_, dim=-1)

                c= lambda x: Encoder(obs_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(obs_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True)).flatten(start_dim=0,end_dim=1)
                class_indices_ = class_indices_.flatten(start_dim=0,end_dim=1)
                
                loss = loss_func(z_, class_indices_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, labels_, class_indices_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for obs_val, labels_val in tqdm(valid_loader):
                    
                    obs_val = obs_val.to(args.device)
                    labels_val = labels_val.to(args.device)

                    class_indices_val = torch.argmax(labels_val, dim=-1)
    
                    c= lambda x: Encoder(obs_val).to(args.device)
                    y_0 =  Encoder(obs_val)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val).flatten(start_dim=0,end_dim=1)
                    class_indices_val = class_indices_val.flatten(start_dim=0,end_dim=1)
                         
                    loss_validation = loss_func(z_val, class_indices_val)
                        
                    del labels_val, z_val, obs_val, class_indices_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            for obs_test, labels_test in tqdm(test_loader):
                
                obs_test = obs_test.to(args.device)
                labels_test = labels_test.to(args.device)

                class_indices_test = torch.argmax(labels_test, dim=-1)

                c= lambda x: Encoder(obs_test).to(args.device)
                y_0 =  Encoder(obs_test)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                
                z_test = Decoder(z_test).flatten(start_dim=0,end_dim=1)
                class_indices_test = class_indices_test.flatten(start_dim=0,end_dim=1)
                
                _, predicted = torch.max(z_test.data, -1)
                n_samples += labels_test.size(0)
                n_correct += (predicted == class_indices_test).sum().item() 

                error += torch.abs(predicted-class_indices_test).sum().item()
                
                tot_predicted = torch.cat([tot_predicted,predicted], dim=0)
                tot_labels = torch.cat([tot_labels, class_indices_test], dim=0)
                

                del z_test, labels_test, obs_test

            avg_error = error/(args.time_points*n_samples)
            acc = 100.0 * n_correct / (args.time_points*n_samples)
            print(f'Accuracy: {acc} %')
            print(f'Average Error: {avg_error}')
            print(classification_report(to_np(tot_labels), to_np(tot_predicted)))
            print(tot_predicted)
            print('Errors: ',torch.abs(tot_predicted-tot_labels))
            # print(roc_curve(to_np(tot_labels),to_np(tot_predicted)))


def Miyawaki_brain_encoding(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())

    if args.AdamW:
        optimizer = torch.optim.AdamW(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)

    if args.loss_func == 'mse':
        loss_func = torch.nn.functional.mse_loss
    elif args.loss_func == 'rel_mse':
        loss_func = relative_mse_loss
    elif args.loss_func == 'huber':
        loss_func = torch.nn.SmoothL1Loss(beta=0.01)
    elif args.loss_func == 'log_cosh':
        loss_func = log_cosh_loss
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for fmri_, stimuli_ in tqdm(train_loader):
                
                fmri_ = fmri_.to(args.device)
                stimuli_ = stimuli_.to(args.device)

                if args.encode_fMRI:
                    fmri_in = torch.cat([fmri_[...,:args.frames_encoded-1],fmri_[...,-1:]],dim=-1)
                    c= lambda x: Encoder(stimuli_.requires_grad_(True),fmri_in).to(args.device)
                    y_0 =  Encoder(stimuli_.requires_grad_(True),fmri_in)[...,-1:,:].to(args.device)
                else:
                    c= lambda x: Encoder(stimuli_).to(args.device)
                    y_0 =  Encoder(stimuli_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True)).permute(0,2,1)
                
                loss = loss_func(z_, fmri_) #+ args.variance_reg*variance_regularization_loss(z_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, fmri_, z_, stimuli_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for fmri_val, stimuli_val in tqdm(valid_loader):
                    
                    fmri_val = fmri_val.to(args.device)
                    stimuli_val = stimuli_val.to(args.device)
    
                    if args.encode_fMRI:
                        fmri_in = torch.cat([fmri_val[...,:args.frames_encoded-1],fmri_val[...,-1:]],dim=-1)
                        c= lambda x: Encoder(stimuli_val.requires_grad_(True),fmri_in).to(args.device)
                        y_0 =  Encoder(stimuli_val.requires_grad_(True),fmri_in)[...,-1:,:].to(args.device)
                    else:
                        c= lambda x: Encoder(stimuli_val).to(args.device)
                        y_0 =  Encoder(stimuli_val.requires_grad_(True))[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val).permute(0,2,1)
                    
                    loss_validation = loss_func(z_val, fmri_val) #+ args.variance_reg*variance_regularization_loss(z_val)

                    if counter == 0:
                        z_val = z_val[0,:,0]
                        fmri_val = fmri_val[0,:,0]
                        z_val_p, fmri_val_p = to_np(z_val), to_np(fmri_val)
                        
                    del z_val, fmri_val, stimuli_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

                    # z_val_brain = args.masker.inverse_transform(z_val_p)
                    # z_val_brain = threshold_img(z_val_brain, threshold=1e-6, copy=False)
                    # fmri_val_brain = args.masker.inverse_transform(fmri_val_p)
                    # fmri_val_brain = threshold_img(fmri_val_brain, threshold=1e-6, copy=False)
                    # plotting.view_img(z_val_brain, 
                    #                   draw_cross=False,  
                    #                   cmap=plotting.cm.black_red
                    #                  )
                    # plotting.view_img(fmri_val_brain, 
                    #                   draw_cross=False,
                    #                   cmap=plotting.cm.black_red
                    #                  )

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")


            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)


            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            test_loss=0.
            all_test_loss = []
            counter=0
            plots1 = []
            plots2 = []
            plots3 = []

            all_pred = torch.tensor([]).to(device)
            all_fmri = torch.tensor([]).to(device)
            
            for fmri_test, stimuli_test in tqdm(test_loader):
                
                fmri_test = fmri_test.to(args.device)
                stimuli_test = stimuli_test.to(args.device)
                
                if args.encode_fMRI:
                    fmri_in = torch.cat([fmri_test[...,:args.frames_encoded-1],fmri_test[...,-1:]],dim=-1)
                    c= lambda x: Encoder(stimuli_test.requires_grad_(True),fmri_in).to(args.device)
                    y_0 =  Encoder(stimuli_test.requires_grad_(True),fmri_in)[...,-1:,:].to(args.device)
                else:
                    c= lambda x: Encoder(stimuli_test).to(args.device)
                    y_0 =  Encoder(stimuli_test.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                
                z_test = Decoder(z_test).permute(0,2,1)
                
                loss_test = loss_func(z_test,fmri_test)

                all_pred = torch.cat([all_pred,z_test])
                all_fmri = torch.cat([all_fmri,fmri_test])

                test_loss += loss_test.item()
                counter+=1

                if counter%100==0:
                    # diff = to_np(torch.abs(z_test[0,:,0]-fmri_test[0,:,0]))
                    diff = to_np(z_test[0,:,0]-fmri_test[0,:,0])
                    
                    z_test_p = to_np(z_test[0,:,0])
                    fmri_test_p = to_np(fmri_test[0,:,0])
                    
                    z_test_brain = args.masker.inverse_transform(z_test_p)
                    z_test_brain_p = threshold_img(z_test_brain, threshold=1e-6, copy=False)
                    
                    fmri_test_brain = args.masker.inverse_transform(fmri_test_p)
                    fmri_test_brain_p = threshold_img(fmri_test_brain, threshold=1e-6, copy=False)
                    
                    diff = args.masker.inverse_transform(diff)
                    diff = threshold_img(diff, threshold=1e-3, copy=False)
                    
                    plot1 = plotting.view_img(z_test_brain_p, 
                                      draw_cross=False,  
                                      cmap=plotting.cm.black_red
                                     )
                    plot2 = plotting.view_img(fmri_test_brain_p, 
                                      draw_cross=False,
                                      cmap=plotting.cm.black_red
                                     )
                    plot3 = plotting.view_img(diff, 
                                      draw_cross=False,
                                      cmap=plotting.cm.black_red
                                     )
                    plots1.append(plot1)
                    plots2.append(plot2)
                    plots3.append(plot3)

            test_loss /= counter
            all_test_loss.append(test_loss)

            r2 = compute_r2(to_np(all_pred),to_np(all_fmri))
            pearson = compute_pearson(all_pred,all_fmri)

            return all_test_loss, plots1, plots2, plots3, r2, pearson


def Miyawaki_brain_decoding(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.AdamW(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    criterion = torch.nn.CrossEntropyLoss(weight=args.class_weights)
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for fmri_, stimuli_ in tqdm(train_loader):
                
                fmri_ = fmri_.to(args.device)
                stimuli_ = stimuli_.to(args.device)

                c= lambda x: Encoder(fmri_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(fmri_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True))
                loss = criterion(z_.permute(0,3,1,2), stimuli_) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, fmri_, z_, stimuli_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for fmri_val, stimuli_val in tqdm(valid_loader):
                    
                    fmri_val = fmri_val.to(args.device)
                    stimuli_val = stimuli_val.to(args.device)
    
                    c= lambda x: Encoder(fmri_val).to(args.device)
                    y_0 =  Encoder(fmri_val)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val)
                    
                    loss_validation = criterion(z_val.permute(0,3,1,2), stimuli_val) 
                        
                    del z_val, fmri_val, stimuli_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

                    # z_val_brain = args.masker.inverse_transform(z_val_p)
                    # z_val_brain = threshold_img(z_val_brain, threshold=1e-6, copy=False)
                    # fmri_val_brain = args.masker.inverse_transform(fmri_val_p)
                    # fmri_val_brain = threshold_img(fmri_val_brain, threshold=1e-6, copy=False)
                    # plotting.view_img(z_val_brain, 
                    #                   draw_cross=False,  
                    #                   cmap=plotting.cm.black_red
                    #                  )
                    # plotting.view_img(fmri_val_brain, 
                    #                   draw_cross=False,
                    #                   cmap=plotting.cm.black_red
                    #                  )

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            test_loss=0.
            all_test_loss = []
            counter=0

            all_pred = torch.tensor([]).to(device)
            all_stimuli = torch.tensor([]).to(device)

            tot_predicted = torch.tensor([]).to(device)
            tot_labels = torch.tensor([]).to(device)
            
            for fmri_test, stimuli_test in tqdm(test_loader):
                
                fmri_test = fmri_test.to(args.device)
                stimuli_test = stimuli_test.to(args.device)
                
                c= lambda x: Encoder(fmri_test).to(args.device)
                y_0 =  Encoder(fmri_test)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                
                z_test = Decoder(z_test)

                all_pred = torch.cat([all_pred,z_test])
                all_stimuli = torch.cat([all_stimuli,stimuli_test])

                counter+=1

                if counter%100==0:
                    predicted = z_test.argmax(dim=-1)
                    plt.imshow(predicted[0,:,0].view(10,10).cpu())
                    plt.show()
                    plt.imshow(stimuli_test[0,:,0].view(10,10).cpu())
                    plt.show()

            
            pixel_accuracy = (all_pred.argmax(dim=-1) == all_stimuli).float().mean()
            print(classification_report(to_np(all_pred.argmax(dim=-1).flatten()),to_np(all_stimuli.flatten())))
            return pixel_accuracy

def Miyawaki_random_geometric_split(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    criterion = torch.nn.CrossEntropyLoss(weight=args.class_weights)
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for fmri_, stimuli_ in tqdm(train_loader):
                
                fmri_ = fmri_.to(args.device)
                stimuli_ = stimuli_.to(args.device)

                c= lambda x: Encoder(fmri_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(fmri_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True))
                
                loss = criterion(z_.permute(0,3,1,2), stimuli_) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, fmri_, z_, stimuli_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for fmri_val, stimuli_val in tqdm(valid_loader):
                    
                    fmri_val = fmri_val.to(args.device)
                    stimuli_val = stimuli_val.to(args.device)
    
                    c= lambda x: Encoder(fmri_val).to(args.device)
                    y_0 =  Encoder(fmri_val)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val)
                    
                    loss_validation = criterion(z_val.permute(0,3,1,2), stimuli_val) 
                        
                    del z_val, fmri_val, stimuli_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

                    # z_val_brain = args.masker.inverse_transform(z_val_p)
                    # z_val_brain = threshold_img(z_val_brain, threshold=1e-6, copy=False)
                    # fmri_val_brain = args.masker.inverse_transform(fmri_val_p)
                    # fmri_val_brain = threshold_img(fmri_val_brain, threshold=1e-6, copy=False)
                    # plotting.view_img(z_val_brain, 
                    #                   draw_cross=False,  
                    #                   cmap=plotting.cm.black_red
                    #                  )
                    # plotting.view_img(fmri_val_brain, 
                    #                   draw_cross=False,
                    #                   cmap=plotting.cm.black_red
                    #                  )

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            # model.eval()
            # Encoder.eval()
            model.train()
            Encoder.train()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            test_loss=0.
            all_test_loss = []
            counter=0

            all_embeddings = torch.tensor([]).to(device)
            
            for fmri_test, stimuli_test in tqdm(test_loader):
                
                fmri_test = fmri_test.to(args.device)
                stimuli_test = stimuli_test.to(args.device)
                
                c= lambda x: Encoder(fmri_test).to(args.device)
                y_0 =  Encoder(fmri_test)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                all_embeddings = torch.cat([all_embeddings,z_test])

                counter+=1

            return all_embeddings


def Huxby_brain_embedding(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()

        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for obs_, labels_ in tqdm(train_loader):
                
                obs_ = obs_.to(args.device)
                labels_ = labels_.to(args.device)

                class_indices_ = torch.argmax(labels_, dim=-1)

                noisy_inputs = obs_+torch.randn_like(obs_)*args.sigma

                c= lambda x: Encoder(noisy_inputs.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(noisy_inputs.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True))
                
                loss = F.mse_loss(z_, obs_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, z_, labels_, class_indices_, noisy_inputs

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for obs_val, labels_val in tqdm(valid_loader):
                    
                    obs_val = obs_val.to(args.device)
                    labels_val = labels_val.to(args.device)

                    class_indices_val = torch.argmax(labels_val, dim=-1)

                    noisy_inputs = obs_val+torch.randn_like(obs_val)*args.sigma
    
                    c= lambda x: Encoder(noisy_inputs).to(args.device)
                    y_0 =  Encoder(noisy_inputs)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val)
                         
                    loss_validation = F.mse_loss(z_val, obs_val)
                        
                    del labels_val, z_val, obs_val, class_indices_val, noisy_inputs

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.train()
            Encoder.train()
            # Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            all_embeddings = torch.tensor([]).to(device)

            for obs_test, labels_test in tqdm(test_loader):
                
                obs_test = obs_test.to(args.device)
                labels_test = labels_test.to(args.device)

                class_indices_test = torch.argmax(labels_test, dim=-1)

                noisy_inputs = obs_test+torch.randn_like(obs_test)*args.sigma

                c= lambda x: Encoder(noisy_inputs).to(args.device)
                y_0 =  Encoder(noisy_inputs)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                
                all_embeddings = torch.cat([all_embeddings,z_test])
                
                
                

            return all_embeddings

            
def Brain_fMRI(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    criterion = torch.nn.MSELoss()
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for fmri_, inputs_ in tqdm(train_loader):
                
                fmri_ = fmri_.to(args.device)
                inputs_ = inputs_.to(args.device)

                c= lambda x: Encoder(inputs_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(inputs_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True))
                
                loss = criterion(z_, fmri_) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, fmri_, z_, inputs_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for fmri_val, inputs_val in tqdm(valid_loader):
                    
                    fmri_val = fmri_val.to(args.device)
                    inputs_val = inputs_val.to(args.device)
    
                    c= lambda x: Encoder(inputs_val).to(args.device)
                    y_0 =  Encoder(inputs_val)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val)
                    
                    loss_validation = criterion(z_val, fmri_val) 
                        
                    del z_val, fmri_val, inputs_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

                    # z_val_brain = args.masker.inverse_transform(z_val_p)
                    # z_val_brain = threshold_img(z_val_brain, threshold=1e-6, copy=False)
                    # fmri_val_brain = args.masker.inverse_transform(fmri_val_p)
                    # fmri_val_brain = threshold_img(fmri_val_brain, threshold=1e-6, copy=False)
                    # plotting.view_img(z_val_brain, 
                    #                   draw_cross=False,  
                    #                   cmap=plotting.cm.black_red
                    #                  )
                    # plotting.view_img(fmri_val_brain, 
                    #                   draw_cross=False,
                    #                   cmap=plotting.cm.black_red
                    #                  )

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            test_loss=0.
            all_test_loss = []
            counter=0

            all_embeddings = torch.tensor([]).to(device)
            all_preds = torch.tensor([]).to(device)
            
            for fmri_test, inputs_test in tqdm(test_loader):
                
                fmri_test = fmri_test.to(args.device)
                inputs_test = inputs_test.to(args.device)
                
                c= lambda x: Encoder(inputs_test).to(args.device)
                y_0 =  Encoder(inputs_test)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                all_embeddings = torch.cat([all_embeddings,z_test])

                z_test = Decoder(z_test)
                
                loss_test = criterion(z_test, fmri_test)

                all_preds = torch.cat([all_preds,z_test])

                test_loss += loss_test.item()
                all_test_loss.append(test_loss)
                counter+=1

                del loss_test

            test_loss /= counter

            return all_preds, all_embeddings, all_test_loss, test_loss


def Riken(model, Encoder, Decoder, dataloaders,args): # 
    
    #metadata for saving checkpoints
    str_model_name = "brain"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    
    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else:
            counter_exp = 0
            for i in txt:
                if i != '.ipynb_checkpoints':
                    counter_exp+=1
            #num_experiments = [int(i[3:]) for i in txt]
            #num_experiments = np.array(num_experiments).max()
            num_experiments = counter_exp
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)


    All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor
            )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        optimizer, scheduler, model, Encoder, Decoder =\
        load_checkpoint(path, optimizer, scheduler, model, Encoder, Decoder)

    spatial_domain = torch.linspace(-1,1,args.space)
    
    if args.mode=='train':
        early_stopping = EarlyStopping(patience=args.patience,min_delta=0)

        all_train_loss=[]
        all_val_loss=[]
        
        train_loader = dataloaders['train']
        valid_loader = dataloaders['valid']
        
        # Train Neural IE

        save_best_model = SaveBestModel()
        start = time.time()
            
        for i in range(args.epochs):
            
            
            model.train()
            Encoder.train()
            Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
                
            for fmri_, fmri_perturbed_ in tqdm(train_loader):
                
                fmri_ = fmri_.to(args.device)
                fmri_perturbed_ = fmri_perturbed_.to(args.device)

                c= lambda x: Encoder(fmri_perturbed_.requires_grad_(True)).to(args.device)
                y_0 =  Encoder(fmri_perturbed_.requires_grad_(True))[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                # z_ = F.softmax(Decoder(z_[...,-1:,:].requires_grad_(True)),dim=-1)
                z_ = Decoder(z_.requires_grad_(True))

                loss = F.mse_loss(z_, fmri_) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, fmri_, z_, fmri_perturbed_

            ## Validating
                
            model.eval()
            Encoder.eval()
            Decoder.eval()
                
            with torch.no_grad():
                
                val_loss = 0.0
                counter = 0
                for fmri_val, fmri_perturbed_val in tqdm(valid_loader):
                    
                    fmri_val = fmri_val.to(args.device)
                    fmri_perturbed_val = fmri_perturbed_val.to(args.device)
    
                    c= lambda x: Encoder(fmri_perturbed_val).to(args.device)
                    y_0 =  Encoder(fmri_perturbed_val)[...,-1:,:].to(args.device)
                    y_init = c(1)
                    
                    if args.ts_integration is not None:
                        times_integration = args.ts_integration
                    else:
                        times_integration = torch.linspace(0,1,args.time_points)
                    
                    z_val = Integral_spatial_attention_solver_multbatch(
                                        times_integration.to(args.device),
                                        y_0.to(args.device),
                                        y_init=y_init,
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=args.mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                        spatial_domain_dim=1,
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        accumulate_grads=False,
                                        initialization=True
                                        ).solve()
    
                    # z_val = F.softmax(Decoder(z_val[...,-1:,:]),dim=-1)
                    z_val = Decoder(z_val)
                    
                    loss_validation = F.mse_loss(z_val, fmri_val) 
                        
                    del z_val, fmri_val, fmri_perturbed_val

                    counter += 1
                    val_loss += loss_validation.item()
                    
                    del loss_validation

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    plt.plot(np.log10(all_train_loss),label='Train loss',color='green')
                    plt.plot(np.log10(all_val_loss),label='Val loss',color='red')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    plt.close()

                    # z_val_brain = args.masker.inverse_transform(z_val_p)
                    # z_val_brain = threshold_img(z_val_brain, threshold=1e-6, copy=False)
                    # fmri_val_brain = args.masker.inverse_transform(fmri_val_p)
                    # fmri_val_brain = threshold_img(fmri_val_brain, threshold=1e-6, copy=False)
                    # plotting.view_img(z_val_brain, 
                    #                   draw_cross=False,  
                    #                   cmap=plotting.cm.black_red
                    #                  )
                    # plotting.view_img(fmri_val_brain, 
                    #                   draw_cross=False,
                    #                   cmap=plotting.cm.black_red
                    #                  )

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, Encoder, Decoder)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break
                
        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')

        test_loader = dataloaders['test']

        tot_predicted = torch.tensor([]).to(device)
        tot_labels = torch.tensor([]).to(device)
        error = 0.
        
        with torch.no_grad():
                
            # model.eval()
            # Encoder.eval()
            model.train()
            Encoder.train()
            Decoder.eval()
                
            n_correct = 0
            n_samples = 0

            test_loss=0.
            all_test_loss = []
            counter=0

            all_embeddings = torch.tensor([]).to(device)
            
            for fmri_test, fmri_perturbed_test in tqdm(test_loader):
                
                fmri_test = fmri_test.to(args.device)
                fmri_perturbed_test = fmri_perturbed_test.to(args.device)
                
                c= lambda x: Encoder(fmri_perturbed_test).to(args.device)
                y_0 =  Encoder(fmri_perturbed_test)[...,-1:,:].to(args.device)
                y_init = c(1)
                
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=args.mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.meshgrid(\
                                            [torch.linspace(-1,1,args.shapes[1]) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=True
                                    ).solve()

                all_embeddings = torch.cat([all_embeddings,z_test])

                counter+=1

            return all_embeddings
