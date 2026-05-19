from torch import nn
import torch
import numpy as np
import math

from IE_source.Galerkin_transformer import SimpleTransformerEncoderLayer, SimpleTransformerEncoderLastLayer
from IE_source.Attentional_IE_solver import interval_function

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)


class kernel_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(kernel_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y

class G_global(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(G_global, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y = y.squeeze()
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y
    

    
class F_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
     
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y
    

class model_blocks(nn.Module):
    def __init__(self,dimension,dim_emb,n_head, n_blocks,n_ff, attention_type, dim_out=2, Final_block = False,dropout=0.1,lower_bound=None,upper_bound=None):
        super(model_blocks, self).__init__()
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.first = nn.Linear(dimension+1,dim_emb)
        self.blocks = nn.ModuleList([SimpleTransformerEncoderLayer(
                                 d_model=dim_emb,n_head=n_head,
                                 dim_feedforward=n_ff,
                                 attention_type=attention_type,
                                 dropout=dropout) for i in range(n_blocks)])
        self.Final_block = Final_block
        if self.Final_block is True:
            self.last_block = SimpleTransformerEncoderLastLayer(
                                    d_model=dim_emb,n_head=n_head,
                                    dim_out=dim_out,dim_feedforward=n_ff,
                                    attention_type=attention_type,
                                    dropout=dropout)
        else:
            self.last_block = nn.Linear(dim_emb,dimension+1)#SimpleTransformerEncoderLayer(d_model=dim_emb,n_head=n_head,attention_type=attention_type,dim_feedforward=n_ff)
        
    def forward(self, x, dynamical_mask=None):
        
        x = self.first.forward(x)
        for block in self.blocks:
            x = block.forward(x,dynamical_mask=dynamical_mask) 
        if self.Final_block is True:
            x = self.last_block.forward(x,dynamical_mask=dynamical_mask)
        else:
            x = self.last_block.forward(x)

        return x

 
def flatten_parameters(NN_F):
    p_shapes = []
    flat_parameters = []
    for p in NN_F.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)

##From Neural ODE paper https://arxiv.org/abs/1806.07366
class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)

        
class Decoder_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.flatten(start_dim=1,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)

        return y_out
    
    
class BrainConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1 = (16,16,10),
                 K2 = (16,16,10),
                 S1 = (8,7,2),
                 S2 = (3,2,1)):
        super(BrainConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K1,
                                     stride=S1
                                    )
        

        self.conv_layer2 = nn.Conv3d(hidden_dim, out_dim,
                                         kernel_size=K2,
                                         stride=S2
                                        )
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        x = x.permute(0,4,1,2,3)
        out = self.conv_layer1(x)
        
        out = self.conv_layer2(out)
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out 

class BrainConvNeuralNet_3D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 times=2,
                 K1 = (16,16,10),
                 K2 = (16,16,10),
                 S1 = (8,7,2),
                 S2 = (3,2,1)):
        super(BrainConvNeuralNet_3D, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K1,
                                     stride=S1
                                    )
        

        self.conv_layer2 = nn.Conv3d(hidden_dim, out_dim,
                                         kernel_size=K2,
                                         stride=S2
                                        )
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.times=times
    
    # Progresses data across layers    
    def forward(self, x):
        x = x.squeeze(-2)
        x = x.permute(0,4,1,2,3)
        out = self.conv_layer1(x)
        
        out = self.conv_layer2(out)
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out.unsqueeze(-2).repeat(1,1,1,1,self.times,1)


class Brain_encoder(nn.Module): 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 times=2,
                 K1 = (16),
                 K2 = (16),
                 S1 = (4),
                 S2 = (4)):
        super(Brain_encoder, self).__init__()
        
        self.conv_layer1 = nn.Conv1d(dim, hidden_dim,
                                     kernel_size=K1,
                                     stride=S1
                                    )
        

        self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                         kernel_size=K2,
                                         stride=S2
                                        )
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.times=times
    
    # Progresses data across layers    
    def forward(self, x):
        output = torch.tensor([]).to(device)
        for i in range(self.times):
            x_in = x[...,i,:]
            x_ = x_in.permute(0,2,1)
            out = self.conv_layer1(x_)
            out = self.relu(out)
            out = self.conv_layer2(out)
            
            out = out.permute(0,2,1)
            
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            output = torch.cat([output,out.unsqueeze(-2)],dim=-2)
        
        return output 


class Decoder_selfsup(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_selfsup, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.flatten(start_dim=1,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)

        return y_out


class Decoder_NN_2D(nn.Module):
    def __init__(self,in_dim,shapes,out_shapes,NL=nn.ELU):
        super(Decoder_NN_2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = math.prod(out_shapes)
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], self.out_dim)
        self.out_shapes = out_shapes
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.permute(0,3,1,2,4)
        y_in = y_in.flatten(-4,-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)
        y = y_out.view(y_out.shape[0],
                   self.out_shapes[0],
                   self.out_shapes[1],
                   self.out_shapes[2],
                   self.out_shapes[3])
        return y

class Decoder_Huxby(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_Huxby, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y = y.permute(0,2,1,3)
        y_in = y.flatten(start_dim=2,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)

        return y_out

class Decoder_Miyawaki(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,size=100,classes=2,NL=nn.ELU):
        super(Decoder_Miyawaki, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim*classes)
        self.size = size
        self.classes = classes
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y = y.permute(0,2,3,1)
        y_in = y.flatten(start_dim=2,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)\
        .view(y_in.shape[0],y_in.shape[1],self.size,self.classes)\
        .permute(0,2,1,3)
        
        return y_out


class StimuliEncoder2D(nn.Module):
    def __init__(self, input_stimuli=100, time_steps=10, output_voxels=5000,
                 input_channels=1, target_channels=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32,
                      kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=(5, 5), stride=(2, 1),
                               padding=(2, 2), output_padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=(5, 5), stride=(2, 1),
                               padding=(2, 2), output_padding=(1, 0)),
            nn.ReLU(inplace=True),
        )

        self.final_fc = nn.Linear(input_stimuli, output_voxels * target_channels)
        self.output_voxels = output_voxels
        self.target_channels = target_channels

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)       
        x = self.decoder(x)       
        x = x.mean(dim=1)       
        batch, stim, t = x.shape
        x = x.permute(0, 2, 1).reshape(batch * t, stim)
        x = self.final_fc(x)      
        x = x.view(batch, t, self.output_voxels, self.target_channels)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x


class Encoder_Miyawaki(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,channels=16,NL=nn.ELU):
        super(Encoder_Miyawaki, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim*channels)
        self.channels = channels
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y = y.permute(0,2,3,1)
        y_in = y.flatten(start_dim=2,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)\
        .view(y_in.shape[0],y_in.shape[1],self.out_dim,self.channels)\
        .permute(0,2,1,3).contiguous()
        
        return y_out

class Stimulus2FMRI(nn.Module):
    def __init__(self,
                 in_channels=1,     # black-&-white → 1 channel
                 embedding_dim=256, 
                 output_voxels=5000,
                 time_steps=3    
                 ):
        super().__init__()

        # Spatial encoder: maps (C=1,H=100,W=100) → embedding_dim
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels,  32, kernel_size=5, padding=2, stride=2),  # → 32×50×50
            nn.ReLU(inplace=True),
            nn.Conv2d(32,           64, kernel_size=5, padding=2, stride=2),  # → 64×25×25
            nn.ReLU(inplace=True),
            nn.Conv2d(64,          128, kernel_size=5, padding=2, stride=2),  # →128×13×13
            nn.ReLU(inplace=True),
            nn.Flatten(),                                                    # →128*13*13
            nn.Linear(128*13*13, embedding_dim),
            nn.ReLU(inplace=True),
        )

        if time_steps is not None:
            self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim,
                              batch_first=True)

        # Final regression head: embedding → fMRI voxels
        self.fc = nn.Linear(embedding_dim, output_voxels)

    def forward(self, stimuli):
        """
        stimuli: either
          - [B, 1, H, W]                (single frame)
          - [B, T, 1, H, W]             (video of T frames)
        returns:
          - [B, output_voxels]          (static)
          - [B, T, output_voxels]       (per-frame, if time_steps given)
        """
        if stimuli.dim() == 4:
            # single-frame case
            emb = self.convnet(stimuli)            # [B, embedding_dim]
            out = self.fc(emb)                     # [B, output_voxels]
            return out

        elif stimuli.dim() == 5:
            B, T, C, H, W = stimuli.shape
            # flatten time into batch for CNN pass
            x = stimuli.view(B*T, C, H, W)         # [B*T,1,H,W]
            emb = self.convnet(x)                  # [B*T, embedding_dim]
            emb = emb.view(B, T, -1)               # [B, T, embedding_dim]

            # pass through GRU
            emb, _ = self.gru(emb)                 # [B, T, embedding_dim]

            # map to voxels for each time
            out = self.fc(emb)                     # [B, T, output_voxels]
            return out

        else:
            raise ValueError("Expected 4D or 5D stimuli, got shape " + str(stimuli.shape))


class Decoder_Huxby_embedding(nn.Module):
    def __init__(self,in_dim,shapes,out_shapes,NL=nn.ELU):
        super(Decoder_Huxby_embedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = math.prod(out_shapes)
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], self.out_dim)
        self.out_shapes = out_shapes
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.flatten(1,-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)
        y = y_out.view(y_out.shape[0],
                   self.out_shapes[0],
                   self.out_shapes[1],
                   self.out_shapes[2])
        return y

class Decoder_fMRI(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,channels=1,NL=nn.ELU):
        super(Decoder_fMRI, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], channels*out_dim)
        self.channels = channels
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y = y.permute(0,2,3,1)
        y_in = y.flatten(start_dim=2,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)\
        .view(y_in.shape[0],y.shape[1],self.out_dim,self.channels)\
        .permute(0,2,1,3)
        
        return y_out

class FMRIFrameEncoder(nn.Module):
    def __init__(self, voxels: int, cond_dim: int, fmri_channels: int = 1, hidden: int = 512):
        super().__init__()
        self.voxels = voxels
        self.fmri_channels = fmri_channels
        in_dim = voxels * fmri_channels

        self.frame_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cond_dim),
            nn.GELU(),
        )

    def forward(self, fmri_frames: torch.Tensor) -> torch.Tensor:
        if fmri_frames.dim() == 3:
            B, V, K = fmri_frames.shape
            x = fmri_frames.unsqueeze(-1)
            C = 1
        elif fmri_frames.dim() == 4:
            B, V, K, C = fmri_frames.shape
            x = fmri_frames

        x = x.reshape(B * K, V * C)      
        h = self.frame_mlp(x)            
        h = h.reshape(B, K, -1)          
        return h.mean(dim=1)             


class Stimulus2FMRIConditionedEncoder4D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,          
        voxels: int = 5438,                
        fmri_channels: int = 1,            
        cond_dim: int = 128,               
        encoded_voxel: int = 512,          
        hidden_channels: int = 64,          
        use_gru: bool = False,
        out_T: int = 25,
        # frames_encoded: int = 2,
    ):
        super().__init__()
        self.encoded_voxel = encoded_voxel
        self.hidden_channels = hidden_channels
        self.latent_dim = encoded_voxel * hidden_channels
        self.out_T = out_T
        # self.frames_encoded = frames_encoded
        self.embedding_dim = embedding_dim

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(6656, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.fmri_encoder = FMRIFrameEncoder(
            voxels=voxels, cond_dim=cond_dim, fmri_channels=fmri_channels
        )

        self.cond_to_film = nn.Sequential(
            nn.Linear(cond_dim, 2 * embedding_dim),
            nn.GELU(),
        )

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                batch_first=True,
            )

        self.to_latent = nn.Linear(embedding_dim, self.out_T*self.latent_dim)

    def forward(self, stimuli: torch.Tensor, fmri_frames: torch.Tensor) -> torch.Tensor:
        cond = self.fmri_encoder(fmri_frames)          
        film = self.cond_to_film(cond)                
        # gamma, beta = film.chunk(self.frames_encoded, dim=-1)

        B = film.shape[0]
        D = self.embedding_dim

        gamma, beta = film.chunk(2, dim=-1)
        # print(gamma.shape,beta.shape)

        def apply_film(emb_seq: torch.Tensor) -> torch.Tensor:
            return emb_seq * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        if stimuli.dim() == 4:
            
            emb = self.convnet(stimuli.permute(0,3,1,2))               
            emb = emb.unsqueeze(1)  
        elif stimuli.dim() == 5:
            B, T, C, H, W = stimuli.shape
            x = stimuli.view(B * T, C, H, W)
            emb = self.convnet(x).view(B, T, -1)      
        else:
            raise ValueError(f"Expected 4D or 5D stimuli, got {stimuli.shape}")
        
        emb = apply_film(emb)                        
        if self.use_gru:
            emb, _ = self.gru(emb)                    

        z = self.to_latent(emb)                    
        z = z.view(z.shape[0], self.encoded_voxel, self.out_T, self.hidden_channels)          
        return z
