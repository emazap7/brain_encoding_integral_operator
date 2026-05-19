import os
import shutil
import torch
import numpy as np
import pickle
import scipy
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import r2_score
# import yaml


# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
from torch.utils.data import Dataset
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from IE_source.integrators import MonteCarlo
mc = MonteCarlo()

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     # if is_best:
#     #     shutil.copyfile(filename, 'model_best.pth.tar')


def to_np(x):
    return x.detach().cpu().numpy()


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_dict_template():
	return {"observed_data": None,
			"observed_tp": None,
			"data_to_predict": None,
			"tp_to_predict": None,
			"observed_mask": None,
			"mask_predicted_data": None,
			"labels": None
			}

def normalize_data(data):
	reshaped = data.reshape(-1, data.size(-1))

	att_min = torch.min(reshaped, 0)[0]
	att_max = torch.max(reshaped, 0)[0]

	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / att_max
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm, att_min, att_max

def display_video(frames, framerate, filename=None):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    
    if filename is not None: anim.save(filename)
    return HTML(anim.to_html5_video())

# def fix_random_seeds(seed=31):
#     """
#     Fix random seeds.
#     """
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
    
def get_system_definition(name, mode='rb'):
    with open(name, mode=mode) as f:
        return f.read()
    
    
class brain_dataset(Dataset):
    def __init__(self, Data, labels):
        self.Data = Data.float()
        self.labels = labels.to(torch.int64)

    def __getitem__(self, index):
        ID = index
        obs_ = self.Data[ID,...]
        labels_ = self.labels[ID]

        return obs_, labels_
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Data)

class Miyawaki_dataset(Dataset):
    def __init__(self, Data, labels):
        self.Data = Data.float()
        self.labels = labels

    def __getitem__(self, index):
        ID = index
        obs_ = self.Data[ID,...]
        labels_ = self.labels[ID]

        return obs_, labels_
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Data)

class Reconstruction_dataset(Dataset):
    def __init__(self, Data, std=0.5, mask=False, mask_ratio=0.1, mask_value=0.0):
        self.Data = Data.float()
        self.std = std
        self.mask = mask
        if mask == True:
            assert 0 <= mask_ratio < 1
            self.mask_ratio = mask_ratio
            self.mask_value = mask_value

    def __getitem__(self, index):
        ID = index
        obs_ = self.Data[ID,...]
        if self.mask == False:
            obs_perturbed = obs_ + self.std*torch.rand_like(obs_)
    
            return obs_, obs_perturbed
        else: 
            Vox, T, C = obs_.shape
            total_elements = Vox*T*C 
            num_to_mask = int(self.mask_ratio * total_elements)
    
            # Create a flat mask: 1 means keep, 0 means mask
            flat_mask = torch.ones(total_elements, device=obs_.device)
            indices = torch.randperm(total_elements, device=obs_.device)[:num_to_mask]
            flat_mask[indices] = 0
    
            # Reshape the mask to match video shape
            mask = flat_mask.view(Vox,T,C)
    
            # Apply the mask
            masked_obs = obs_ * mask + self.mask_value * (1 - mask)
    
            return obs_, masked_obs
            
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Data)
    
    
class LRScheduler():

    def __init__(
        self, optimizer, patience=100, min_lr=1e-9, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
    
    def get_last_lr(self):
        last_lr = self.lr_scheduler.get_last_lr()
        return last_lr
        
        
class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, path, current_valid_loss, epoch, model, kernel, Encoder, Decoder):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            if kernel is not None: kernel_state = {'state_dict': kernel.state_dict()}
            if Encoder is not None: F_func_state = {'state_dict': Encoder.state_dict()}
            if Decoder is not None: f_func_state = {'state_dict': Decoder.state_dict()}
            
            torch.save(model, os.path.join(path,'model.pt'))
            if kernel is not None: torch.save(kernel_state, os.path.join(path,'kernel.pt'))
            if Encoder is not None: torch.save(F_func_state, os.path.join(path,'Encoder.pt'))
            if Decoder is not None: torch.save(f_func_state, os.path.join(path,'Decoder.pt'))
            
            
def load_checkpoint(path, optimizer, scheduler, kernel, Encoder, Decoder):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
     
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    start_epoch = checkpoint['epoch']
    offset = start_epoch
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
     
    checkpoint = torch.load(os.path.join(path, 'kernel.pt'), map_location=map_location)
    kernel.load_state_dict(checkpoint['state_dict'])
     
    checkpoint = torch.load(os.path.join(path, 'Encoder.pt'), map_location=map_location)
    Encoder.load_state_dict(checkpoint['state_dict'])
     
    checkpoint = torch.load(os.path.join(path, 'Decoder.pt'), map_location=map_location)
    Decoder.load_state_dict(checkpoint['state_dict'])
    
    return optimizer, scheduler, kernel, Encoder, Decoder
                
            
class Select_times_function():
    def __init__(self,times,max_index):
        self.max_index = max_index
        self.times = times

    def select_times(self,t):
            values = torch.Tensor([])
            indices = []
            for i in range(1,t.size(0)):
                if t[i]<= self.times[self.max_index-1]:
                    values = torch.cat([values,torch.Tensor([t[i]])])
                    indices += [i]
                else:
                    pass
            return values, indices

def to_np(x):
    return x.detach().cpu().numpy()


def normalization(Data):
    for i in range(Data.size(2)):
        di = Data[:,:,i]/torch.abs(Data[:,:,i]).max()
        di = di.unsqueeze(2)
        if i == 0:
            Data_norm = di
        else:
            Data_norm = torch.cat([Data_norm,di],2)
    return Data_norm

    
def plot_reconstruction(data_to_plot, predicted_to_plot, frames_to_drop, path_to_save_plots, name, epoch, args):
    
    
    # print('data_to_plot.shape: ',data_to_plot.shape)
    num_points_to_plot = 20 if data_to_plot.shape[-1]>20 else data_to_plot.shape[-1]
    
    n_plots_x = 10 #I want 10 images on the horizontal
    n_plots_y = int(np.ceil(num_points_to_plot/n_plots_x))
    # print('n_plots_x: ',n_plots_x)
    # print('n_plots_y: ',n_plots_y)
    if frames_to_drop is not None and frames_to_drop>0:
        extrapolation = data_to_plot.shape[-1]-frames_to_drop
    else: extrapolation = data_to_plot.shape[-1]+1
    
    fig,ax = plt.subplots(int(2*n_plots_y),n_plots_x, figsize=(15,5), facecolor='w',dpi=300)
    c=0
    for idx_row in range (n_plots_y): 
        for idx_col in range(n_plots_x):
            if c < data_to_plot.shape[-1]:
                # print('idx_row,idx_col,c: ',idx_row,idx_col,c)
                ax[2*idx_row,idx_col].imshow(data_to_plot[:,:,c],vmin=args.range_imshow[0],vmax=args.range_imshow[1],
                                             aspect='auto')
                ax[2*idx_row,idx_col].axis('off')
                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[:,:,c].flatten(), predicted_to_plot[:,:,c].flatten())
                if c>=extrapolation:
                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2), color= 'green', fontweight='bold')
                else: ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[:,:,c],vmin=args.range_imshow[0],vmax=args.range_imshow[1],
                                              aspect='auto')
                ax[2*idx_row+1,idx_col].axis('off')
                c+=1
            ax[2*idx_row+1,idx_col].axis('off')
            ax[2*idx_row,idx_col].axis('off')
    fig.tight_layout()
    if args.mode=='train':
        plt.savefig(os.path.join(path_to_save_plots, name +str(epoch)))
        plt.close('all')
    else:
        plt.show()
    
    del data_to_plot, predicted_to_plot, frames_to_drop
    
    return
  

def sampling_grid(data, patch_sampling_cfg, frame_size, patch_size,current_epoch=None):
# def sampling_grid(data, hparams, frame_size, patch_size):
    # -- sample patches in grid-like form
    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"]
    mode = patch_sampling_cfg["mode"]
#     grid_rows = frame_size//patch_size

    if data.shape[-1]> num_patches:
        rep_param = False
    else:
        rep_param = True
    
#     # Gutavo's method
#     T = torch.from_numpy(np.repeat(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param)), grid_rows**2)) # Select T samples with 
# #     T = torch.from_numpy(np.repeat(np.arange(grid_rows**2), num_patches))    
#     P_row = torch.from_numpy(np.tile(np.tile(np.arange(grid_rows) * patch_size, grid_rows), num_patches))
#     P_col = torch.from_numpy(np.tile(np.repeat(np.arange(grid_rows) * patch_size, grid_rows), num_patches))
# #     print('Gustavo:')
# #     print('T: ',T)
# #     print('P_row: ',P_row)
# #     print('P_col: ',P_col)
# #     print('T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))

#     # To sample a grid from the frame
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param))) # Select T samples with 
#     P_row = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#     P_col = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
# #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
#     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
#     T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()

    if mode=="inference":
#         if current_epoch%2==0:
            # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
#         else:
#             T = torch.arange(1,data.shape[0]-1,step=2) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
#         P_row = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#         P_col = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
            
        
    else: 
        # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[-1]-1), num_patches-1, replace=rep_param))) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[-1]-1]))) # Append the last frame manually 
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
    
    return T, P_row, P_col

    
def plot_dim_vs_time(obs_to_print, time_to_print, z_to_print, dummy_times_to_print, z_all_to_print, frames_to_drop, path_to_save_plots, name, epoch, args):
    # obs_to_print[0,:], time_to_print[0,:], z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:]
    verbose=False
    # obs_ = obs_[:-frames_to_drop]
    # ts_ = ts_[:-frames_to_drop]
    if verbose: 
        print('[plot_dim_vs_time] obs_to_print.shape: ',obs_to_print.shape)
        print('[plot_dim_vs_time] time_to_print.shape: ',time_to_print.shape)
        print('[plot_dim_vs_time] args.num_dim_plot: ',args.num_dim_plot)
        print('[plot_dim_vs_time] dummy_times_to_print.shape: ',dummy_times_to_print.shape)
        print('[plot_dim_vs_time] z_all_to_print.shape: ',z_all_to_print.shape)
        
        
    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
    ax=ax.ravel()
    for idx in range(args.num_dim_plot):
        # ax[idx].plot(time_to_print,z_to_print[:,idx],c='r', label='model')
        ax[idx].plot(dummy_times_to_print,z_all_to_print[:,idx],c='r', label='model')
        # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
        if frames_to_drop is not None and frames_to_drop>0:
            ax[idx].scatter(time_to_print[:-frames_to_drop],obs_to_print[:-frames_to_drop,idx],label='Data',c='blue', alpha=0.5)
            ax[idx].scatter(time_to_print[-frames_to_drop:],obs_to_print[-frames_to_drop:,idx],label='Hidden',c='green', alpha=0.5)
        else:
            ax[idx].scatter(time_to_print[:],obs_to_print[:,idx],label='Data',c='blue', alpha=0.5)
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("dim"+str(idx))
        #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
        ax[idx].legend()
        # timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.tight_layout()

    if args.mode=='train' or path_to_save_plots is not None:
        plt.savefig(os.path.join(path_to_save_plots, name + str(epoch)))
        plt.close('all')
    else: plt.show()
    
    del obs_to_print, time_to_print, z_to_print, frames_to_drop
    
class SaveLastState:
    ''' have to be redone. It shoudl be using torch.save'''
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    def __call__(self, path, current_valid_loss, epoch, model, G_NN = None, kernel=None, F_func = None, f_func=None):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest training loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            torch.save(model, os.path.join(path,'model_train.pt'))
            
            
def relative_mse_loss(y_pred, y_true, epsilon=1e-8):
    numerator = torch.norm(y_pred - y_true, p=2) ** 2
    denominator = torch.norm(y_true, p=2) ** 2 + epsilon
    return numerator / denominator


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=('relu2_2',), weights=None):
        """
        Perceptual loss using a pretrained VGG network.
        - layers: tuple of strings indicating which layers' features to use.
        - weights: dictionary mapping layer names to weights for the loss contribution.
        """
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.eval()  # Set to evaluation mode
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.layers = layers
        if weights is None:
            self.weights = {layer: 1.0 for layer in layers}
        else:
            self.weights = weights
        
        # Mapping layer names to indices in VGG16 features
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }
    
    def forward(self, input, target):
        """
        Compute the perceptual loss between input and target.
        Both input and target should be normalized appropriately (e.g., using ImageNet stats).
        """
        loss = 0.0
        x = input
        y = target
        for name, module in self.vgg._modules.items():
            x = module(x)
            y = module(y)
            # Check if the current layer index corresponds to one we care about
            for layer in self.layers:
                if int(name) == self.layer_name_mapping[layer]:
                    loss += self.weights[layer] * nn.functional.mse_loss(x, y)
        return loss

# # Example usage:
# if __name__ == "__main__":
#     # Assume we have two images of shape (B, C, H, W) normalized for VGG (e.g., using ImageNet normalization)
#     B, C, H, W = 4, 3, 224, 224
#     img1 = torch.rand(B, C, H, W)
#     img2 = torch.rand(B, C, H, W)
    
#     perceptual_loss_fn = VGGPerceptualLoss(layers=('relu2_2', 'relu3_3'), weights={'relu2_2': 1.0, 'relu3_3': 1.0})
#     p_loss = perceptual_loss_fn(img1, img2)
#     print("Perceptual Loss:", p_loss.item())


def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true) + 1e-12))

def variance_regularization_loss(y_pred):
    return torch.exp(-torch.mean(torch.var(y_pred, dim=0)))

def compute_r2(y_pred, y_true):
    batch_size, voxels, timepoints = y_pred.shape
    r2_scores = []
    
    for v in range(voxels):
        r2 = r2_score(y_true[:, v, :].flatten(), y_pred[:, v, :].flatten())
        r2_scores.append(r2)
    
    return torch.tensor(r2_scores)

def compute_pearson(y_pred, y_true):
    y_pred_mean = torch.mean(y_pred, dim=2, keepdim=True)
    y_true_mean = torch.mean(y_true, dim=2, keepdim=True)
    
    numerator = torch.sum((y_pred - y_pred_mean) * (y_true - y_true_mean), dim=2)
    denominator = torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2, dim=2) * 
                             torch.sum((y_true - y_true_mean) ** 2, dim=2) + 1e-6)
    
    return torch.mean(numerator / denominator, dim=0)
	
