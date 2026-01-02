"""
Modified version of the original code: https://github.com/yewsiang/ConceptBottleneck


"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
#from inception_model import MLP, inception_v3, End2EndModel
from  torchvision.models import inception_v3,Inception_V3_Weights


# Independent & Sequential Model
class ModelXtoC(nn.Module):
    def __init__(self,pretrained, freeze,use_aux, n_concepts):
        """
        Model used for the X -> C part of the Independent and Sequential models

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            freeze (bool): If True, freezes the weights of the model except for the last layer
            use_aux (bool): If True, uses the auxilary model for the inception model
            n_concepts (int): Number of concepts to predict
        
        """

        super(ModelXtoC, self).__init__()

        self.use_aux = use_aux
        
        num_classes = n_concepts # number of classes is the number of concepts

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None, aux_logits=use_aux)# Load the inception model

        self.model.fc = nn.Linear(2048, num_classes) # Change the last layer to output the number of concepts

        self.model.AuxLogits.fc = nn.Linear(768, num_classes) # Change the last layer in the auxilary model to output the number of concepts

        if freeze:  # only finetune fc layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False

    def forward(self, x):

        if self.use_aux and self.training:
            Chat, aux_Chat = self.model(x)

            return Chat, aux_Chat
        
        else:
            Chat = self.model(x)

            return Chat


class ModelCtoY(nn.Module):
    """
    Simple one-layer MLP model for the C -> Y
    """
    def __init__(self, input_dim, num_classes):
        super(ModelCtoY, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x



# Joint Model
class ModelXtoCtoY(nn.Module):
    

    def __init__(self, pretrained, freeze, n_classes, use_aux, n_concepts,use_sigmoid=False):
        """
        Model used for the X -> C -> Y part of the Joint model

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            freeze (bool): If True, freezes the weights of the model except for the last layer
            n_classes (int): Number of classes to predict
            use_aux (bool): If True, uses the auxilary model for the inception model
            n_concepts (int): Number of concepts to predict
        """

        super(ModelXtoCtoY, self).__init__()

        self.use_sigmoid = use_sigmoid

        self.sailency_output = False #ToDo Remove this

        self.use_aux = use_aux

        self.XtoC_model = ModelXtoC(pretrained, freeze,use_aux, n_concepts)

        self.XtoC_model.fc = nn.Linear(2048, n_concepts) # Change the last layer to output the number of concepts

        self.CtoY_model = ModelCtoY(n_concepts, n_classes) # Create the MLP model for the C -> Y

        if use_aux: #Make and auxilary model for the C -> Y
            #self.XtoC_model.AuxLogits.fc = nn.Linear(768, n_concepts)
            self.aux_CtoY_model = ModelCtoY(n_concepts, n_classes)
        
        if freeze:  # only finetune fc layer
            for name, param in self.XtoC_model.named_parameters():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the model. Returns the Chat and Yhat unless the sailency_output is set to either 'C' or 'Y' in which case only the Chat or Yhat is returned
        """

        if self.use_aux and self.training:
            Chat, aux_Chat = self.XtoC_model(x)



            #Apply sigmoid activation function before final layer
            if self.use_sigmoid: 
                Yhat = torch.sigmoid(self.CtoY_model(torch.sigmoid(Chat)))
                aux_Yhat = torch.sigmoid(self.aux_CtoY_model(torch.sigmoid(aux_Chat)))
            else:
                Yhat = self.CtoY_model(Chat)
                aux_Yhat = self.aux_CtoY_model(aux_Chat)
            
            return Chat, Yhat, aux_Chat, aux_Yhat
        
        else:
            Chat = self.XtoC_model(x)

            #Apply sigmoid activation function before final layer
            if self.use_sigmoid: 
                Yhat = torch.sigmoid(self.CtoY_model(torch.sigmoid(Chat)))
            else:
                Yhat = self.CtoY_model(Chat)


            if not self.sailency_output: #If we are not interested in the sailency map
                return Chat, Yhat

            else:
                # Only return the output we want to create the sailency map of
                if self.sailency_output == 'C':
                    return Chat
                elif self.sailency_output == 'Y':
                    return Yhat 
                
    def set_sailency_output(self, output):
        """
        Function for returning the sailency map of the model
        args:
            output (str): Either 'C' or 'Y' for the sailency map of the Chat or Yhat
        """

        if output in ['C', 'Y']:
            self.sailency_output = output
        else:
            raise ValueError("Invalid output type for sailency map. Please provide one of the following: C, Y")
        


# Standard Model
class ModelXtoY(nn.Module):
    def __init__(self,pretrained, freeze,use_aux, n_classes):
        """
        Model used for the X -> Y just a inception Standard model

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            freeze (bool): If True, freezes the weights of the model except for the last layer
        
        Returns:
            model (nn.Module): Inception v3 model
        """
        super(ModelXtoY, self).__init__()

        self.use_aux = use_aux

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else False, aux_logits=use_aux)# Load the inception model

        self.model.fc = nn.Linear(2048, n_classes) # Change the last layer to output the number of classes

        self.model.AuxLogits.fc = nn.Linear(768, n_classes) # Change the last layer in the auxilary model to output the number of classes

        self.activation = nn.Softmax(dim=1) # Use softmax activation function


        if freeze:  # only finetune fc layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False

    def forward(self, x):

        if self.use_aux and self.training:
            Yhat, aux_Yhat = self.model(x)
            #Yhat = self.activation(Yhat)
            #aux_Yhat = self.activation(aux_Yhat)
            return Yhat, aux_Yhat
        
        else:
            Yhat = self.model(x)
            #Yhat = self.activation(Yhat)
            return Yhat


def get_inception_transform(mode="train",methode="original",resol=299):
    """
    Get the transform for the inception model.

    Note: The CUB dataset in the original paper used a random resized crop.
    This method would most likly cut away some concepts thus making the classifier predict things it can not see. 

    Args:
    mode: str, either 'train' or 'val'
    methode: str, either 'original', 'center' or 'resize'


    Returns: torchvision.transforms.Compose object
    """


    if methode == "RandomResize":
        
        if mode == "train":
            transform = transforms.Compose([
                #transforms.Resize((resized_resol, resized_resol)),
                #transforms.RandomSizedCrop(resol),
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
                ])
        else:
            transform = transforms.Compose([
                #transforms.Resize((resized_resol, resized_resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
                ])
    elif methode == "CenterCrop":
        #Apply center crop to both train and val
        if mode == "tain":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.CenterCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
    
    elif methode == "Resize":
        #Apply resize to both train and val
        if mode == "train":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.Resize((resol, resol)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
        else:
            transform = transforms.Compose([
                #transforms.Resize((resol, resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
    
    else:
        raise ValueError("Invalid methode for transform. Please provide one of the following: RandomResize, CenterCrop, Resize")

    return transform