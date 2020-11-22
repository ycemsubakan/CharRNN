import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
from tqdm import tqdm
#from helpers import *
#from model import *
#from generate import *
import pdb
import numpy as np
import torch.utils.data as data_utils
import time
import torch.nn.functional as F



class CharRNN(nn.Module):
    """
    This class holds the model definition for the CharRNN, as well as the functions for:
    training -- trainer() 
    data generation -- generate_data()
    data loader construction -- get_loader()
    """
    def __init__(self, L, K1, K2, n_layers, dropout):
        super(CharRNN, self).__init__()

        # set the model parameters
        self.L = L
        self.K1 = K1
        self.K2 = K2
        self.n_layers = n_layers

        # set the model layers
        self.embedder = nn.Embedding(L, K1)
        self.rnn = nn.LSTM(K1, K2, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(K2, L)


    def forward(self, inp, h0=None, gen_mode=False):
        '''
        # inp : input data 
        # h0 : initial hidden state (only needed if we are calling forward in gen_mode)
        # gen_mode : should be set to True if the is used to generate data, else it is set to False. 
        '''

        # get the embedding
        code = self.embedder(inp)

        # predict the RNN latent states
        if gen_mode:
            h, hs = self.rnn(code, h0)
        else:
            h, hs = self.rnn(code)

        # pass the RNN embeddings through the output layer
        out = self.out(h)

        return out, hs

    def trainer(self, arguments, train_loader, EP):
        # the trainer function

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.train()
        # instantiate the loss
        lossf = nn.CrossEntropyLoss()
        # iterate over training epochs
        for ep in range(EP):
            # iterate over the batches
            for i, (tar) in enumerate(train_loader):
                opt.zero_grad()
               
                tar = tar[0].to(arguments.device)
               
                # Note!!! the model tries to predict tar[:, 1:] given tar[:, :-1]
                xhat, _ = self.forward(tar[:, :-1])
               
                #tar_rsh = tar.contiguous().view(-1).float()
                # compute the loss 
                cost = lossf(xhat.reshape(-1, xhat.size(-1)),
                             tar[:, 1:].reshape(-1))

                # compute the gradient
                cost.backward()

                # take a training step
                opt.step()

                # spit out the current loss and other things on the screen
                print('EP [{}/{}], batch [{}/{}], \
                       Cost is {}, Learning rate is {}'.format(ep+1, EP, i+1,
                                                         len(train_loader),
                                                         cost.item(),
                                                         opt.param_groups[0]['lr']))

            ### print the reconstructions
            recons = xhat.argmax(dim=2)[0]

            recons_chars = ''.join([all_characters[ind] for ind in recons])
            print(recons_chars)


       
        return xhat
                                             
               

    def generate_data(self, N, L, arguments):
        # this function generates a sequence with the trained model
        # N is the length of the sequence to be generated 
        # L is the number of possible characters
        # arguments passes the command line arguments into this function
        self.eval()
       
        inp = torch.randint(L, (1,)).to(arguments.device)
        outs = []
        all_probs = []
        for l in range(N):
            inp = inp.unsqueeze(0)
            if l == 0:
                out, h = self.forward(inp, gen_mode=True)
            else:
                out, h = self.forward(inp, h, gen_mode=True)
            
            probs = F.softmax(out.squeeze(), dim=-1)
            all_probs.append(probs.data.unsqueeze(0))

            inp = torch.multinomial(probs, 1)
           
            outs.append(inp.item())
        return outs, torch.cat(all_probs, dim=0)


def partition_text_file(file, chunk_len, pct_train):
    '''
    preprocessing function to create text segments for training
    '''
    chunks = []
    tmp = ''
    for idx, c in enumerate(file):
        tmp += c
        if idx % chunk_len == 0:
            chunks.append(tmp)
            tmp = ''
    np.random.shuffle(chunks)
    train_chunks = chunks[0 : int(pct_train * len(chunks))]
    test_chunks = chunks[int(pct_train * len(chunks)) :]
    train_file = ''.join(train_chunks)
    test_file = ''.join(test_chunks)
    return train_file, test_file


def get_loader(fl, chunk_len, batch_size):
    """
    function to create the data loader 
    """
    
    inputs = fl
    #targets = fl[1:]

    N = len(inputs)

    N = N - N % chunk_len
    inputs = inputs[:N]
   
    all_inputs = [char_tensor(inputs[i:i+chunk_len]).view(1,-1) for i in range(0, N, chunk_len)]
    all_inputs = torch.cat(all_inputs, 0)

    dataset = data_utils.TensorDataset(all_inputs)

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return loader



def save():
    save_filename = os.path.splitext(os.path.basename(arguments.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)



def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def char_tensor(string):
    # Turning a string into a tensor
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor


def time_since(since):
    # Readable time elapsed
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)




"""
# if you want to use a new file 

file, file_len = read_file(arguments.filename)
train_file, test_file = partition_text_file(file, arguments.chunk_len, 0.75)
train_file, vld_file = partition_text_file(train_file, arguments.chunk_len, 0.75)
pickle.dump(train_file, open('train_set.pk', 'wb'))
pickle.dump(vld_file, open('vld_set.pk', 'wb'))
pickle.dump(test_file, open('test_set.pk', 'wb'))
"""



if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filename', type=str, default='')
    argparser.add_argument('--n_epochs', type=int, default=300, help='number of training epochs')
    argparser.add_argument('--K1', type=int, default=100, help='dimensionality of the embeddings')
    argparser.add_argument('--K2', type=int, default=100, help='dimensionality of the latent states of the rnn')

    argparser.add_argument('--num_layers', type=int, default=1, help='number of rnn layers')
    argparser.add_argument('--cuda', type=int, default=1)
    argparser.add_argument('--seed', type=int, default=1, help='random number seed')
    argparser.add_argument('--chunk_len', type=int, default=100, help='length of the segments to train on')
    argparser.add_argument('--batch_size', type=int, default=100)
    argparser.add_argument('--dropout', type=float, default=0.1, help='droput keep probability')


    arguments = argparser.parse_args()

    arguments.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.cuda:
        torch.cuda.manual_seed(arguments.seed)


    timestamp = time.time()

    # load the training file
    train_file = pickle.load(open('train_set.pk','rb'))
    all_characters = list(set(train_file))
    n_characters = len(all_characters)

    # get the loader 
    train_loader = get_loader(train_file, arguments.chunk_len, arguments.batch_size)

    # instantiate the model and get it in the device
    RNN = CharRNN(L=n_characters, K1=arguments.K1, K2=arguments.K2, n_layers=arguments.num_layers,
                  dropout=arguments.dropout)
    RNN = RNN.to(arguments.device)

    # train the model 
    RNN.trainer(arguments, train_loader, EP=arguments.n_epochs)

    # generate the data 
    gen_data, _ = RNN.generate_data(N=600, L=n_characters, arguments=arguments)

    # reformat the data as a string
    gen_chars = ''.join([all_characters[ind] for ind in gen_data])
    print(gen_chars)




