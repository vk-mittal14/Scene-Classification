# import the lib. we wil need
from torch import nn
import torch
import torchvision
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


class Encoder(nn.Module):
    """ 
    Takes a image and returns an encoded reprsentation of that image. 
    """

    def __init__(self, enc_img_size):
        """
        Initializes the Encoder Layer 
        Params: 
            enc_img_size: the size of the encoded image
                         after passing thourgh the CNN Model
        Returns: 
            output of shape (batch_size, encode_out_num_channels, enc_img_size*enc_img_size)
        """
        super(Encoder, self).__init__()
        self.enc_img_size = enc_img_size
        arch = 'resnet50'

        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        self.num_channels = 2048

        # remove the last pool and fc layer 
        modules = list(model.children())[:-2]
        
        # remake the network 
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(enc_img_size)


    def forward(self, images):
        """
        Does a forwad pass over the encoder model
        Params: 
            images = images from the dataset
        """
        out = self.resnet(images)
        out = self.pool(out)
        encoder_out = out.view(-1, self.num_channels, self.enc_img_size*self.enc_img_size)
        # (batch_size, encode_out_num_channels, enc_img_size*enc_img_size)
        return encoder_out


class Attention(nn.Module):
    """
    Attention Mask over the encoded image produced by the CNN. 

    It takes the input of the LSTM Cell state to update its values & 
    encoded image.
    We will use a hidden Layer to transform the output of
    the LSTM to the dimension the encoded image.
    After that softmax will be applied on that vector. 
    After that element wise product of the image and the attention vector
    will be taken. This vector will be passed to the LSTM again. 
    """

    def __init__(self, num_channels, encoder_dim, decoder_dim):
        """
        Initialize the Attention Layer. 
        Args:
            num_channels -> number of channels in the output of the CNN model
            encoder_dim -> the height or width dimension of the output of the CNN Model (num_channels, H, W). 
            decoder_dim -> the dimension of the output of the decoder.
        """
        super(Attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_channels = num_channels
        # linear layer to transform decoder's output to attention dimension
        self.decoder_att = nn.Linear(decoder_dim, encoder_dim*encoder_dim)
        # softmax layer over the output of the above linear layer
        self.softmax = nn.Softmax(dim=1) 
        # avg. pool the output after applying the softmax over it
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Does a forward pass.
        Args: 
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
            decoder_hidden -> the output of the decoder (batch_size, decoder_dim)
        Returns: 
            attention_weighted_encoding (batch_size, encode_out_num_channels)
        """

        # (batch_size, encoder_dim*encoder_dim)
        decoder2att = self.decoder_att(decoder_hidden)
        # (batch_size, encoder_dim*encoder_dim)
        attn = self.softmax(decoder2att)

        # calculate the attention over the image
        # (batch_size, encode_out_num_channels,  encoder_dim*encoder_dim)
        attention_weighted_encoding = encoder_out*(attn.unsqueeze(1))
        # Reshape to (batch_size, encode_out_num_channels, encoder_dim, encoder_dim)
        attention_weighted_encoding = attention_weighted_encoding.view(-1, self.num_channels, self.encoder_dim, self.encoder_dim)
        # apply Adaptive Avg. Pool 
        pooled = self.avg_pool(attention_weighted_encoding) # (batch_size, encode_out_num_channels, 1, 1)
        attention_weighted_encoding = pooled.view(-1, self.num_channels) # (batch_size, encode_out_num_channels)
        return attention_weighted_encoding


class Decoder(nn.Module):
    """
    This will recieve the input from the Attention Layer & encoder,
    that will be passed on to the LSTM cell to do it's work.
    """

    def __init__(self, num_channels, encoder_dim, decoder_dim1, decoder_dim2, decoder_dim3, num_classes, num_lstm_cell):
        """
        Args: 
        num_channels -> num_channels in the output of the CNN model  
        encoder_dim ->  the height or width dimension of the output of the CNN Model (num_channels, H, W).
        decoder_dim1 -> the dim. of the decoder1
        decoder_dim2 -> the dim. of the decoder2 
        decoder_dim3 -> the dim. of the decoder3 
        num_lstm_cell -> the recurrence number
        num_classes -> the number of classes
        """
        super(Decoder, self).__init__()

        self.num_classes = num_classes
        self.num_lstm_cell = num_lstm_cell

        self.attention = Attention(num_channels, encoder_dim, decoder_dim1)
        self.decode_step1 = nn.LSTMCell(num_channels, decoder_dim1, bias=True)
        self.decode_step2 = nn.LSTMCell(decoder_dim1, decoder_dim2, bias=True)
        self.decode_step3 = nn.LSTMCell(decoder_dim2, decoder_dim3, bias=True)
        self.last_linear = nn.Linear(decoder_dim3, num_classes)
        self.softmax = nn.Softmax(dim = 1)

        self.init_h1 = nn.Linear(encoder_dim*encoder_dim, decoder_dim1)  # linear layer to find initial hidden state of LSTMCell-1
        self.init_c1 = nn.Linear(encoder_dim*encoder_dim, decoder_dim1) 
        self.init_h2 = nn.Linear(encoder_dim*encoder_dim, decoder_dim2)  # linear layer to find initial hidden state of LSTMCell-2
        self.init_c2 = nn.Linear(encoder_dim*encoder_dim, decoder_dim2) 
        self.init_h3 = nn.Linear(encoder_dim*encoder_dim, decoder_dim3)  # linear layer to find initial hidden state of LSTMCell-3
        self.init_c3 = nn.Linear(encoder_dim*encoder_dim, decoder_dim3) 
        

    def init_hidden_state(self, encoder_out):
        """
        Initializes the param of h, c for 3 stacked LSTM 
        Args: 
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h1 = self.init_h1(mean_encoder_out)  # (batch_size, decoder_dim)
        c1 = self.init_c1(mean_encoder_out)

        h2 = self.init_h2(mean_encoder_out)  # (batch_size, decoder_dim)
        c2 = self.init_c2(mean_encoder_out)

        h3 = self.init_h3(mean_encoder_out)  # (batch_size, decoder_dim)
        c3 = self.init_c3(mean_encoder_out)
        return h1, c1, h2, c2, h3, c3

    def forward(self, encoder_out):
        """
        Does a forward pass.
        Args:
            encoder_out -> the output of the encoder (batch_size, encode_out_num_channels, encoder_dim*encoder_dim)
        """
        
        batch_size = encoder_out.size(0)
        h1, c1, h2, c2, h3, c3  = self.init_hidden_state(encoder_out)

        # stores the output of the LSTM Cells (We are having num_lstm_cell number of cells)
        y_complete = torch.zeros(size = (self.num_lstm_cell, batch_size, self.num_classes))

        for i in range(self.num_lstm_cell):
            attention_weighted_encoding = self.attention(encoder_out, h1) # (batch_size, encode_out_num_channels)
            h1, c1 = self.decode_step1(attention_weighted_encoding, (h1, c1)) # (batch_size, decoder_dim1), # (batch_size, decoder_dim1)
            h2, c2 = self.decode_step2(h1, (h2, c2)) # (batch_size, decoder_dim2), # (batch_size, decoder_dim2)  
            h3, c3 = self.decode_step3(h2, (h3, c3)) # (batch_size, decoder_dim3), # (batch_size, decoder_dim3)
            out = self.last_linear(h3) # (batch_size, num_classes)
            y_t = self.softmax(out) # (batch_size, num_classes)
            y_complete[i] = y_t

        return y_complete.sum(dim =0 )