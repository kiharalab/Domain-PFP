import numpy as np
import torch
torch.manual_seed(2)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os



class DomainGOEmbeddingModel(torch.nn.Module):
    """
    Domain GO Embedding Model
    """

    def __init__(self, domain_mapper, go_mapper, emb_dim=256, lmbd=0.1, n_neuron_1 = 128, dp1=0.05, name='mdl'):
        """
        Initializes the model

        Args:
            domain_mapper (dict): python dictionary mapping Domain to id.
            go_mapper (dict): python dictionary mapping GO term to id.
            emb_dim (int): dimension of Embedding.
            lmbd (float): lambda in L1 regularization. Defaults to 0.1.
            n_neuron_1 (int, optional): number of neurons in dense layer. Defaults to 128.
            dp1 (float, optional): dropout rate. Defaults to 0.05.
            name (str, optional): name of model. Defaults to 'mdl'.
        """
        super(DomainGOEmbeddingModel, self).__init__()

        self.name = name        

        self.emb_dim = emb_dim
        self.lmbd = lmbd

        self.num_domains = len(domain_mapper)
        self.domain_mapper = domain_mapper
        self.embedding_domain = torch.nn.Embedding(num_embeddings=self.num_domains, embedding_dim=self.emb_dim)

        self.num_go = len(go_mapper)
        self.go_mapper = go_mapper
        self.embedding_go = torch.nn.Embedding(num_embeddings=self.num_go, embedding_dim=self.emb_dim)   
            
        self.dense1 = torch.nn.Linear(emb_dim, n_neuron_1)        
        self.act1 = torch.nn.ReLU()
        self.dp1 = torch.nn.Dropout(dp1)

        self.out = torch.nn.Linear(n_neuron_1,1)


    def forward(self, domain_id, go_id):
        """
        Implements forward pass

        Args:
            domain_id (int): id of the domain
            go_id (int): id of the GO term

        Returns:
            _type_: _description_
        """

        domain_embedding = self.embedding_domain(domain_id)
        go_embedding = self.embedding_go(go_id)
                
        feat = torch.mul(domain_embedding, go_embedding)     

        feat = self.dense1(feat)
        feat = self.act1(feat)
        feat = self.dp1(feat)

        out = self.out(feat)

        return out, torch.mean(torch.abs(domain_embedding))     # return the score and absolute value of embeddings layer for L1 regularization


    def get_domain_embedding(self, domain_id):
        """
        returns the embedding of a domain

        Args:
            domain_id (int): id of the domain

        Returns:
            tensor: domain embedding
        """

        dmn_embdng = self.embedding_domain(domain_id)
        
        return dmn_embdng

    def get_go_embedding(self, go_id):
        """
        returns the embedding of a GO terms

        Args:
            go_id (int): id of the GO term

        Returns:
            tensor: GO term embedding
        """
        go_embdng = self.embedding_go(go_id)
        
        return go_embdng


def train_domain_go_model(mdl, X_domain, X_go, Y, val_split, batch_size=10240, epochs=200, lr=0.001, gpu_device=0):
    """
    Train the Domain GO Embedding Model

    Args:
        mdl (torch model): Domain GO Embedding Model
        X_domain (numpy array): array of domain indices
        X_go (numpy array):  array of GO terms indices
        Y (numpy array): array of p(GO|domain) scores
        val_split (float): validation split
        batch_size (int, optional): number of samples in a batch. Defaults to 10240.
        epochs (int, optional): number of epochs. Defaults to 200.
        lr (float, optional): learning rate. Defaults to 0.001.
        gpu_device (int, optional): index of gpu. Defaults to 0.
    """

    device = None       # initialize device

    if torch.cuda.is_available():       # GPU 
        device = torch.device('cuda')
        torch.cuda.set_device(gpu_device)
        print(f'Using GPU {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:                               # CPU                               
        device = torch.device('cpu')
        print('Using CPU')
        
    mdl.to(device)

    filename = mdl.name
    mdl_pth = os.path.join('saved_models',filename)


    criterion = torch.nn.MSELoss()          # Loss function
    mae_metric = torch.nn.L1Loss()          # Evaluation Metric
    optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)       # Optimizer

    i_train, i_val, _, _ = train_test_split(np.arange(len(X_domain)), np.arange(len(X_domain)), test_size=val_split, random_state=2)        # train-val split

    X_domain_train = X_domain[i_train]              # splitting the data
    X_domain_val = X_domain[i_val]
    X_go_train = X_go[i_train]
    X_go_val = X_go[i_val]
    Y_train = Y[i_train]
    Y_val = Y[i_val]

    best_loss_trn = 1000000000          # initialize 
    best_loss_val = 1000000000
    last_best_epoch = -1
                                        # convert data to tensor

    X_domain_val = torch.tensor(X_domain_val).to(torch.int64)       
    X_go_val = torch.tensor(X_go_val).to(torch.int64)   
    Y_val = torch.tensor(Y_val).to(torch.float32)
    
    X_domain_val = X_domain_val.to(device)
    X_go_val = X_go_val.to(device)
    Y_val = Y_val.to(device)


    for epoch in tqdm(range(epochs)): 

        cur_loss_trn = 0
        cur_loss_val = 0
        cur_metric_trn = 0 
        cur_metric_val = 0

        indices = np.arange(len(X_domain_train))            
        np.random.shuffle(indices)                          # shuffle training data

        for i in range(0, len(indices), batch_size):

            minibatch = indices[i:min(i+batch_size,len(indices))]           # create mini-batch

            X_domain_minibatch = X_domain_train[minibatch]
            X_go_minibatch = X_go_train[minibatch]
            Y_minibatch = Y_train[minibatch]

            X_domain_minibatch = torch.tensor(X_domain_minibatch).to(torch.int64)
            X_go_minibatch = torch.tensor(X_go_minibatch).to(torch.int64)
            Y_minibatch = torch.tensor(Y_minibatch).to(torch.float32)

            X_domain_minibatch = X_domain_minibatch.to(device)
            X_go_minibatch = X_go_minibatch.to(device)
            Y_minibatch = Y_minibatch.to(device)


            optimizer.zero_grad()

            outputs = mdl(X_domain_minibatch, X_go_minibatch)                            
            _preds = outputs[0].to(torch.float32).flatten()
            _embds = outputs[1].to(torch.float32).flatten() 

            loss_part1 = criterion(_preds, Y_minibatch)
            l1_reg = mdl.lmbd * torch.mean(torch.abs(_embds))
            loss = loss_part1 + l1_reg 
                
            metric = mae_metric(_preds, Y_minibatch)
                        
            loss.backward()
            optimizer.step()

            # print statistics
            cur_loss_trn += loss.item()*(len(minibatch)/len(Y_train))
            cur_metric_trn += metric.item() *(len(minibatch)/len(Y_train))

        with torch.no_grad():

            indices_val = np.arange(len(X_domain_val))

            for i in range(0, len(indices_val), batch_size):

                minibatch_val = indices_val[i:min(i+batch_size,len(indices_val))]

                X_domain_val_minibatch = X_domain_val[minibatch_val]
                X_go_val_minibatch = X_go_val[minibatch_val]
                Y_val_minibatch = Y_val[minibatch_val]

                X_domain_val_minibatch = torch.tensor(X_domain_val_minibatch).to(torch.int64)
                X_go_val_minibatch = torch.tensor(X_go_val_minibatch).to(torch.int64)
                Y_val_minibatch = torch.tensor(Y_val_minibatch).to(torch.float32)

                X_domain_val_minibatch = X_domain_val_minibatch.to(device)
                X_go_val_minibatch = X_go_val_minibatch.to(device)
                Y_val_minibatch = Y_val_minibatch.to(device)

                val_outputs = mdl(X_domain_val_minibatch, X_go_val_minibatch)

                _val_preds = val_outputs[0].to(torch.float32).flatten()
                _val_embds = val_outputs[1].to(torch.float32).flatten() 

                val_loss_part1 = criterion(_val_preds, Y_val_minibatch)
                val_l1_reg = mdl.lmbd * torch.mean(torch.abs(_val_embds))
                val_loss = criterion(_val_preds, Y_val_minibatch)
                    
                val_metric = mae_metric(_val_preds, Y_val_minibatch)
                
                cur_loss_val += val_loss.item()*(len(minibatch_val)/len(Y_val))
                cur_metric_val += val_metric.item()*(len(minibatch_val)/len(Y_val))
        

        tqdm.write(f'Epoch {epoch + 1} : loss {cur_loss_trn:.3f} val_loss {cur_loss_val:.3f} mae {cur_metric_trn:.3f} val_mae {cur_metric_val:.3f}')


        if(cur_loss_trn < best_loss_trn):
            tqdm.write(f'Training loss improved from {best_loss_trn:.3f} to {cur_loss_trn:.3f}')
            best_loss_trn = cur_loss_trn
        else:
            tqdm.write(f'Training loss didn\'t improve from {best_loss_trn:.3f}')

        if(cur_loss_val < best_loss_val):
            tqdm.write(f'Validation loss improved from {best_loss_val:.3f} to {cur_loss_val:.3f}')
            best_loss_val = cur_loss_val
            last_best_epoch = epoch
            torch.save(mdl.state_dict(), mdl_pth)
        else:
            tqdm.write(f'Validation loss didn\'t improve from {best_loss_val:.3f} at epoch {last_best_epoch}')


def load_domaingo_embedding_model_weights(mdl, mdl_weight_pth):
    """
    loads the weights of a domaingo embedding model

    Args:
        mdl (torch model): model
        mdl_weight_pth (string): path to the saved weights

    Returns:
        torch model: model with the loaded weights
    """
    

    # since the model is computationally less expensive, we load it in cpu
    mdl.load_state_dict(torch.load(mdl_weight_pth,map_location=torch.device('cpu')))
    mdl.eval()

    return mdl
