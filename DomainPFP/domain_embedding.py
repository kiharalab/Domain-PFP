import torch
torch.manual_seed(0)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

DOUBLE_PRECISION = 12


class DomainEmbedding:
    """
    Domain Embedding Object
    """
    def __init__(self, domaingo_mdl, domain_mapper):
        """
        Initialization

        Args:
            domaingo_mdl (torch model): Domain GO Embedding Model
            domain_mapper (dict): maps domain to id
            
        """
        self.domain_mapper = domain_mapper
        self.total_domain = len(domain_mapper)

                            # extract the domain embeddings from the model
        self.embedding = np.array(domaingo_mdl.get_domain_embedding(torch.tensor(np.arange(self.total_domain))).detach().numpy(),dtype=np.float64)

        self.embedding = np.round(self.embedding,DOUBLE_PRECISION)

        
                                        # Normalize domain embeddings
        scaler = MinMaxScaler()        
        self.embedding = scaler.fit_transform(self.embedding)
        self.embedding = np.round(self.embedding,DOUBLE_PRECISION)

        
    def get_embedding(self, inp):
        """
        returns the embedding of the input domain

        Args:
            inp (int): id of the input domain
                or
            inp (str): the input domain

        Returns:
            numpy array: embedding of the domain
        """

        if isinstance(inp,str):
            if inp in self.domain_mapper:
                return np.array(self.embedding[self.domain_mapper[inp]])
            else:                         # if domain not found, return 0 default
                return np.array(self.embedding[0]*0)
        
        else:
            if inp<0:                               # dummy value -1 can be used to obtain default 0 embedding
                return np.array(self.embedding[0]*0)

            if inp<len(self.domain_mapper):
                return np.array(self.embedding[inp])

            else:                         # if domain not found, return 0 default
                return np.array(self.embedding[0]*0)


    def contains(self, inp):
        """
        check if a particular domain is in the model

        Args:
            inp (int): id of the input domain
                or
            inp (str): the input domain

        Returns:
            bool: domain exists or not
        """
        if isinstance(inp,str):
            return inp in self.domain_mapper
        else:
            return inp < len(self.embedding)
        
