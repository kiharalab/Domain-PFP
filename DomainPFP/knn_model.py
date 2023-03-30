import numpy as np
from tqdm import tqdm
import random
from scipy.spatial.distance import cdist


DOUBLE_PRECISION = 12

class NearestNeighbors:
    """
    Nearest Neighbor search implementation
    """

    def __init__(self, n_neighbors, metric='euclidean',p=2):
        """
        Initialization

        Args:
            n_neighbors (int): number of neighbors
            metric (str, optional): distance metric. Defaults to 'euclidean'.
            p (int, optional): Defaults to 2.
        """
        
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.features = []
        self.labels = []

    def fit(self,X_train,Y_train):
        """
        Train the model

        Args:
            X_train (list or numpy array): protein embeddings
            Y_train (list or numpy array): list of GO terms or protein ids
        """

        self.features = np.array(X_train)
        self.labels = np.array(Y_train)

    def predict(self,X_test):
        """
        Obtain model prediction

        Args:
            X_test (list or numpy array): protein embedding

        Returns:
            numpy array: distance from the other proteins in the embedding space
        """

        dist_mat = cdist(X_test,self.features,metric=self.metric,p=self.p)
        
        return dist_mat 
    
    def kneighbors(self,X_test,nneighs=None,return_distance=True):
        """
        returns the k nearest neighbors, optionally with the distances

        Args:
            X_test (list or numpy array): protein embedding
            nneighs (int, optional): Number of nearest neighbors. Defaults to None.
            return_distance (bool, optional): will distance be returned? Defaults to True.

        Returns:
            dists: distance of the nearest neighbors
            neighs: the nearest neighbors
        """
        

        if(nneighs==None):
            nneighs = self.n_neighbors

        

        dist_mat = self.predict(X_test)

        dists = []
        neighs = []

        for i in range(len(X_test)):

            srted_dist = np.array(dist_mat[i])
            srted_dist.sort()
            if(nneighs==-1):                        # if -1 is given as input, all neighbors are returned
                thresh = 0
            else:
                thresh = srted_dist[nneighs]
            neigh = np.array(np.where(dist_mat[i]<=thresh)[0])
            dist = np.array(dist_mat[i][neigh])

            dists.append(dist)
            neighs.append(neigh)

        if(return_distance):
            return dists,neighs
        else:
            return neighs

    

def prepare_knn_data(all_protein_domains,all_protein_go,all_protein_domains_valid,all_protein_go_valid,all_protein_domains_test,all_protein_go_test, dmn_embedding):
    """
    Prepare data to train the KNN model

    Args:
        all_protein_domains (dict): python dictionary containing the domains of all the proteins in training data
        all_protein_go (dict): python dictionary containing the GO terms of all the proteins in training data
        all_protein_domains_valid (dict): python dictionary containing the domains of all the proteins in validation data
        all_protein_go_valid (dict): python dictionary containing the GO terms of all the proteins in validation data
        all_protein_domains_test (dict): python dictionary containing the domains of all the proteins in test data
        all_protein_go_test (dict): python dictionary containing the GO terms of all the proteins in test data
        dmn_embedding (DomainEmbedding object): Domain Embedding object

    Returns:
        tuple of lists: (X_dmn_embd_train,          => embeddings of proteins in the training data
                         Y_p_id_train,              => ids of the proteins in the training data
                         Y_go_terms_train,          => list of GO terms of proteins in the training data
                         X_dmn_embd_valid,          => embeddings of proteins in the validation data
                         Y_p_id_valid,              => ids of the proteins in the validation data
                         Y_go_terms_valid,          => list of GO terms of proteins in the validation data
                         X_dmn_embd_test,           => embeddings of proteins in the test data
                         Y_p_id_test,               => ids of the proteins in the test data
                         Y_go_terms_test)           => list of GO terms of proteins in the test data
    """


    X_dmn_embd_train = []               # domain embeddings
    Y_p_id_train = []                   # protein name / id
    Y_go_terms_train = []               # GO terms

    X_dmn_embd_valid = []               
    Y_p_id_valid = []
    Y_go_terms_valid = []

    X_dmn_embd_test = []
    Y_p_id_test = []
    Y_go_terms_test = []

    prtns = set(all_protein_domains.keys())
    prtns = prtns.intersection(set(all_protein_go.keys()))
    prtns = list(prtns)
    prtns.sort()

    for prtn in tqdm(prtns):

        if(len(all_protein_domains[prtn])==0):          # if no domain is present
            continue    

        x_dmn_embd = dmn_embedding.get_embedding(-1)            # initialize embedding
        cnt = 0                                                 # initialize number of domains
        p_id = prtn    

        for dmn in (all_protein_domains[prtn]):
            if(dmn_embedding.contains(dmn)):          # if domain is in model
                x_dmn_embd += dmn_embedding.get_embedding(dmn)
                x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)              # ensuring precision
                cnt += 1

        if(cnt>0):                          # if there are domains in this protein
            x_dmn_embd /= cnt

        x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)
        
        X_dmn_embd_train.append(x_dmn_embd)        
        Y_p_id_train.append(p_id)
        Y_go_terms_train.append(all_protein_go[prtn])



    valid_prtns = set(all_protein_domains_valid.keys())
    valid_prtns = valid_prtns.intersection(set(all_protein_go_valid.keys()))
    valid_prtns = list(valid_prtns)
    valid_prtns.sort()

    for prtn in tqdm(valid_prtns):

        x_dmn_embd = dmn_embedding.get_embedding(-1)
        cnt = 0
        p_id = prtn    

        for dmn in all_protein_domains_valid[prtn]:

            if(dmn_embedding.contains(dmn)):          # if domain is in model
                x_dmn_embd += dmn_embedding.get_embedding(dmn)
                x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)
                cnt += 1
            
        if(cnt>0):                          # if there are domains in this protein
            x_dmn_embd /= cnt

        x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)            
        
        X_dmn_embd_valid.append(x_dmn_embd)        
        Y_p_id_valid.append(p_id)
        Y_go_terms_valid.append(all_protein_go_valid[prtn])
        

    test_prtns = list(all_protein_domains_test.keys())
    test_prtns.sort()
    

    for prtn in tqdm(test_prtns):

            x_dmn_embd = dmn_embedding.get_embedding(-1)
            cnt = 0
            p_id = prtn    
            
            for dmn in (all_protein_domains_test[prtn]):

                if(dmn_embedding.contains(dmn)):          # if domain is in model
                    x_dmn_embd += dmn_embedding.get_embedding(dmn)
                    x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)                    
                    cnt += 1
            
            if(cnt>0):                                  # if there are domains in this protein
                x_dmn_embd /= cnt


            x_dmn_embd = np.round(x_dmn_embd,DOUBLE_PRECISION)
            
        
            X_dmn_embd_test.append(x_dmn_embd)        
            Y_p_id_test.append(p_id)
            Y_go_terms_test.append(all_protein_go_test[prtn])

        
    X_dmn_embd_train = np.array(X_dmn_embd_train)
    Y_p_id_train = np.array(Y_p_id_train)
    X_dmn_embd_valid = np.array(X_dmn_embd_valid)
    Y_p_id_valid = np.array(Y_p_id_valid)
    X_dmn_embd_test = np.array(X_dmn_embd_test)
    Y_p_id_test = np.array(Y_p_id_test)

    return (X_dmn_embd_train,Y_p_id_train,Y_go_terms_train,X_dmn_embd_valid,Y_p_id_valid,Y_go_terms_valid,X_dmn_embd_test,Y_p_id_test,Y_go_terms_test)



class Weighted_KNN_Model:
    """
    Implementation of the weighted KNN model
    """

    def __init__(self, n_neighbors):
        """
        Initialzation function

        Args:
            n_neighbors (int): number of neighbors
        """
        np.random.seed(2)               # setting seeds for reproducibility
        random.seed(2)

        self.n_neighbors = n_neighbors
        self.mdl = NearestNeighbors(n_neighbors=n_neighbors,metric='minkowski', p=2)        # Euclidean distance

    def train(self, X_dmn_embd_train, Y_p_id_train):
        """
        Train the model

        Args:
            X_dmn_embd_train (list or numpy array): protein embeddings
            Y_p_id_train (list or numpy array): list of protein ids
        """

        self.mdl.fit(X_dmn_embd_train, Y_p_id_train)
        

    def get_nearest_neighbors(self,inp_dmn_embd):
        """
        returns the k nearest neighbors,  with the distances

        Args:
            inp_dmn_embd (list or numpy array): protein embedding
            
        Returns:
            dists: distance of the nearest neighbors
            neighs: the nearest neighbors
        """        
        dists,neighs = self.mdl.kneighbors([inp_dmn_embd], return_distance=True)
        
        return dists[0],neighs[0]


    def get_nearest_neighbors_batch(self, inp_dmn_embd):
        """
        returns the k nearest neighbors,  with the distances, in batches

        Args:
            inp_dmn_embd (list or numpy array): protein embeddings

        Returns:
            dists: distance of the nearest neighbors
            neighs: the nearest neighbors
        """

        dists,neighs = self.mdl.kneighbors(inp_dmn_embd, self.n_neighbors, return_distance=True)

        return dists,neighs 


    def get_neighbor_go_terms(self,Y_go_terms_train, inp_dmn_embd, min_cnt=None):
        """
        Get the GO terms of the neighbors

        Args:
            Y_go_terms_train (list or numpy array): GO terms for the proteins in the training data
            inp_dmn_embd (numpy array): protein embedding
            min_cnt (int, optional): GO term needs to be present in at least how many proteins. Defaults to None.

        Returns:
            list: GO terms
        """

        if(min_cnt==None):                      # A GO term needs to be present in half of the proteins at least
            min_cnt = self.n_neighbors//2

        dists,neighs  = self.get_nearest_neighbors(inp_dmn_embd)

        go_terms = {}

        for n in neighs:
            for go_trm in Y_go_terms_train[n]:
                if go_trm not in go_terms:
                    go_terms[go_trm] = 0

                go_terms[go_trm] += 1

        out = []

        for go_trm in go_terms:
            if go_terms[go_trm]>=min_cnt:
                out.append(go_trm)

        return out

    def get_neighbor_go_terms_proba(self, Y_go_terms_train, inp_dmn_embd):
        """
        Get the probability scores of the GO terms

        Args:
            Y_go_terms_train (list or numpy array): GO terms for the proteins in the training data
            inp_dmn_embd (numpy array): protein embedding

        Returns:
            dict: Python dictionary containing the GO terms and their probabilities
        """

        dists,neighs  = self.get_nearest_neighbors(inp_dmn_embd)

        normalizer = 0
        for d in dists:
            if(d==0):
                normalizer += 1000          # avoid 1/0
            else:
                normalizer += 1/d

        go_terms = {}

        for i in range(len(neighs)):
            for go_trm in Y_go_terms_train[neighs[i]]:
                if go_trm not in go_terms:
                    go_terms[go_trm] = 0

                if(dists[i]==0):
                    go_terms[go_trm] += 1000        # avoid 1/0
                else:
                    go_terms[go_trm] += 1/dists[i]

        for go_trm in go_terms:
            go_terms[go_trm] /= normalizer
            
        return go_terms

    def get_neighbor_go_terms_proba_batch(self, Y_go_terms_train, inp_dmn_embd):
        """
        Get the probability scores of the GO terms in batches

        Args:
            Y_go_terms_train (list or numpy array): GO terms for the proteins in the training data
            inp_dmn_embd (numpy array): protein embeddings

        Returns:
            dict: Python dictionary containing the GO terms and their probabilities
        """

        dists,neighs = self.get_nearest_neighbors_batch(inp_dmn_embd)

        normalizer = []
        for i in tqdm(range(len(dists))):
            nrm = 0
            for j in range(len(dists[i])):
                if(dists[i][j]==0):
                    nrm = nrm+1000                  # avoid 1/0
                else:
                    nrm = nrm + 1/dists[i][j]
        
            normalizer.append(nrm)


        go_terms = []

        for i in tqdm(range(len(neighs)),desc='computing go terms'):
            go_terms.append({})
            
            for j in range(len(neighs[i])):
                
                for go_trm in Y_go_terms_train[neighs[i][j]]:
                    if go_trm not in go_terms[-1]:
                        go_terms[-1][go_trm] = 0

                    if(dists[i][j]==0):
                        go_terms[-1][go_trm]  += 1000           # avoid 1/0
                    else:
                        go_terms[-1][go_trm] += 1/dists[i][j]


        
        for i in tqdm(range(len(go_terms))):
            for go_trm in go_terms[i]:
                go_terms[i][go_trm] = go_terms[i][go_trm]/normalizer[i]
            
        return go_terms

