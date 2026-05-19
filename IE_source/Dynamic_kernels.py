from torch import nn
import torch
import numpy as np

from IE_source.Galerkin_transformer import SimpleTransformerEncoderLayer, SimpleTransformerEncoderLastLayer
from IE_source.Attentional_IE_solver import interval_function

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)


class ConvNeuralNet_3D(nn.Module):
    '''Default produces [4,4,4] spatial dims. Choice K1=(8,8,2),K2=(8,8,1),S1=(5,5,2),S2=(7,6,1) produces [5,5,5]'''
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1 = (16,16,2),
                 K2 = (16,16,2),
                 S1 = (8,7,2),
                 S2 = (3,2,1)):
        super(ConvNeuralNet_3D, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K1,
                                     stride=S1
                                    )
        

        self.conv_layer2 = nn.Conv3d(hidden_dim, out_dim,
                                         kernel_size=K2,
                                         stride=S2
                                        )
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        #out = out.permute(0,3,1,2)
        return out 

    
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1=[8,8],
                 K2=[8,8],
                 S1=[2,2],
                 S2=[2,2]
                ):
        
        super(ConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K1,
                                    stride=S1
                                    )
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_layer2 = nn.Conv2d(hidden_dim, out_dim,
                                     kernel_size=K2,
                                     stride=S2
                                    )
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,3,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,3,1,2)
        return out   
    
    
    
class Decoder_Dynamic_3D(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,space_dimensions=[208,168,10],NL=nn.ELU):
        super(Decoder_Dynamic_3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.space_dimensions = space_dimensions
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.permute(0,4,1,2,3,5)
        y_in = y_in.flatten(-4,-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)
        y = y_out.permute(0,2,1)
        y = y.view(y_out.shape[0],
                   self.space_dimensions[0],
                   self.space_dimensions[1],
                   self.space_dimensions[2],
                   y_out.shape[1])

        return y
    
    
class Conv2plus1D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1=[8,8],
                 S1=[8,8],
                 time_points=5,
                 total_time=10
                ):
        
        super(Conv2plus1D, self).__init__()
        
        self.conv_layer = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K1,
                                    stride=S1
                                    )
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.time_points = time_points
        self.total_time = total_time
    
    # Progresses data across layers    
    def forward(self, x):
        x = x.permute(0,3,4,1,2)
        out = torch.cat([self.conv_layer(x[:,i,...]).unsqueeze(1) for i in range(self.time_points)],dim=1)
        
        out = out.permute(0,3,4,1,2)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        out = torch.cat([out[...,:self.time_points-1,:],
                         out[...,self.time_points-1:,:]\
                         .repeat(1,1,1,self.total_time-self.time_points+1,1)],dim=-2)
        
        return out  
    
class Single3DConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,K=[4,4,5],S=[4,4,1]):
        super(SingleConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K,
                                     stride=S)
        
 
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,4,1,2,3)
        return out   

    
class Decoder_classification(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_classification, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        
        y_in = y.flatten(1,-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)

        return y_out
    
    
class Conv3plus1D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1=[8,8,3],
                 S1=[8,8,2],
                 time_points=5,
                 total_time=10
                ):
        
        super(Conv3plus1D, self).__init__()
        
        self.conv_layer = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K1,
                                    stride=S1
                                    )
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.time_points = time_points
        self.total_time = total_time
    
    # Progresses data across layers    
    def forward(self, x):
        x = x.permute(0,4,5,1,2,3)
        out = torch.cat([self.conv_layer(x[:,i,...]).unsqueeze(1) for i in range(self.time_points)],dim=1)
        
        out = out.permute(0,3,4,5,1,2)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        out = torch.cat([out[...,:self.time_points-1,:],
                         out[...,self.time_points-1:,:]\
                         .repeat(1,1,1,1,self.total_time-self.time_points+1,1)],dim=-2)
        
        return out 

    
class Conv2D_interpolation(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K=[8,8],
                 S=[8,8],
                 total_time=10
                ):
        
        super(Conv2D_interpolation, self).__init__()
        
        self.conv_layer = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K,
                                    stride=S
                                    )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.total_time = total_time
       
    def forward(self, x):
        x = x.permute(0,3,4,1,2)
        out = torch.cat([self.conv_layer(x[:,i,...]).unsqueeze(1) for i in range(x.shape[1])],dim=1)
        
        out = out.permute(0,3,4,1,2)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        out = torch.nn.functional.interpolate(
            out.permute(0,4,1,2,3),
            size=[out.shape[1],
                  out.shape[2],
                  self.total_time],
            mode='trilinear')
        out = out.permute(0,2,3,4,1)
        
        return out  
