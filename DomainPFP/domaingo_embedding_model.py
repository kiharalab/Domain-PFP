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
