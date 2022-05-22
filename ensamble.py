
from matplotlib.pyplot import cla
from numpy import dtype
import torch
import torch.nn.functional as F
import gc
from torch import nn
from zmq import device
import helpers


class NewEnsamble42(nn.Module):
    
    # define the class constructor
    def __init__(self, numChannels=12, classes=10, preprocessors=None):

        # call super class constructor
        super(NewEnsamble42, self).__init__()
        self.trainable = False
        self.numChannels = numChannels
        self.preprocessors = preprocessors 
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(classes, 20, bias=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(20, classes, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        
    # define network forward pass
    def forward(self, images):
        
        predicted, layers = [], []
        for name, p in self.preprocessors.items():
            preiction = p(images)
            predicted.append(preiction)
            #layers.append(layer)
            #print(layer.shape, preiction.shape)
        context = torch.cat(tuple(predicted), dim=1).cuda()
        #print(context)
        #context = self.sigmoid(context)
        #context = self.linear1(context)
        #context = self.relu(context)
        #context = self.output(context)
        x = self.logsoftmax(context)  
        # return forward pass result
        #x = torch.nn.functional.one_hot(torch.argmax(context, dim=1), 10).float()
        #print(tuple(self.preprocessors.keys())[x[0]])
        return x
    
    
    

# implement OwnStructure
class Ensamble42(nn.Module):
    
    # define the class constructor
    def __init__(self, numChannels=12, classes=10, preprocessors=[]):

        # call super class constructor
        super(Ensamble42, self).__init__()

        self.numChannels = numChannels
        self.preprocessors = [p for p in preprocessors]
        
        self.prep_batch = nn.BatchNorm1d(len(preprocessors))
        
        self.deep_context_batch = nn.BatchNorm1d(len(preprocessors)*256)
        self.deep_context_linear1 = nn.Linear(len(preprocessors)*256, 512)
        self.deep_context_elu1 = nn.ELU()
        self.deep_context_linear2 = nn.Linear(512, 128)
        self.deep_context_elu2 = nn.ELU()
        
        self.context1 = nn.Linear(len(preprocessors) + 128, 10, bias=True)
        self.context_elu = nn.ELU()
        
        self.output = nn.Linear(10, classes, bias=True)
        
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        

    def _conv_layer_set(self, in_c, out_c,):
      conv_layer = nn.Sequential(
          nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=0),
          nn.ELU(),
          nn.MaxPool2d((2,2))
      )
      return conv_layer
    
    # define network forward pass
    def forward(self, images):
        
        predicted, layers = [], []
        for p in self.preprocessors:
            (preiction, layer) = p(images)
            predicted.append(preiction)
            layers.append(layer)
            #print(layer.shape, preiction.shape)
        context = torch.cat(tuple(predicted), dim=1).cuda()
        deep_context = torch.cat(tuple(layers), dim=1).cuda()
        
        
        context = self.prep_batch(context)
        
        deep_context = self.deep_context_batch(deep_context)
        deep_context = self.deep_context_linear1(deep_context)
        deep_context = self.deep_context_elu1(deep_context)
        #rint(deep_context.shape)
        deep_context = self.deep_context_linear2(deep_context)
        deep_context = self.deep_context_elu2(deep_context)
        #print(context.shape, deep_context.shape)
        context = torch.cat((context, deep_context), dim=1)
        
        context = self.context1(context)
        context = self.context_elu(context)
        context = self.output(context)
        
        # define layer 3 forward pass
        x = self.logsoftmax(context)  
        # return forward pass result
        return x

# implement OwnStructure# implement OwnStructure
class Preprocessor42(nn.Module):
    
    # define the class constructor
    def __init__(self, numChannels=12, classes=2, hooked = False):

        # call super class constructor
        super(Preprocessor42, self).__init__()

        self.numChannels = numChannels
        self.hooked = hooked
        self.hidden_output = None
        self.conv_1 = self._conv_layer_set(numChannels, 32)
        self.conv_2 = self._conv_layer_set(32, 64)
        
        
        self.dropout = nn.Dropout(.2)
        
        self.batch1 = nn.BatchNorm1d(64*14**2)
        self.linear1 = nn.Linear(64*14**2, 512)
        self.elu1 = nn.ELU()
        
        self.batch2 = nn.BatchNorm1d(512)
        
        self.linear2 = nn.Linear(512, 256)
        self.elu2 = nn.ELU() 
        
        self.linear3 = nn.Linear(256, classes, bias=True) 
        
        # add a softmax to the last layer
        self.sigmoid = nn.Sigmoid()
        self.linear2.register_forward_hook(self.get_activation())

    def _conv_layer_set(self, in_c, out_c,):
      conv_layer = nn.Sequential(
          nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=0),
          nn.ELU(),
          nn.MaxPool2d((2,2))
      )
      return conv_layer
  
    def get_activation(self):
        def hook(model, input, output):
            self.hidden_output = output.detach()
        return hook
        
        
    # define network forward pass
    def forward(self, images):
        
        # reshape image pixels
        batch_size = images.shape[0]
        
        device = next(self.parameters()).device
        x = self.conv_1(torch.tensor(images, dtype=torch.float32).to(torch.device(device)))
        x = self.conv_2(x)
        
        x = x.view(batch_size, -1)
        x= self.dropout(x)
        
        x = self.batch1(x)
        x= self.linear1(x)
        x= self.elu1(x)
        x = self.batch2(x)
        x = self.linear2(x)
        x= self.elu2(x)
        x= self.linear3(x)
        x = self.sigmoid(x)
        
        return x if not self.hooked else (x, self.hidden_output)
