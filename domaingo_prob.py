from optparse import OptionParser
from tqdm import tqdm
import pickle
import torch
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join('DomainPFP')))
from domaingo_embedding_model import DomainGOEmbeddingModel, load_domaingo_embedding_model_weights



parser = OptionParser(add_help_option=False)

parser.add_option('--domain',  help=  'input InterPro domain')
parser.add_option('--GO',  help='input GO term')


(options, args) = parser.parse_args()


def calc_domaingo_prob(domain,GO):
    """
    Compute and return the prediction score for domain-GO association probability using the SwissProt models

    Args:
        domain (str): the input Domain
        GO (str): the input GO term        
    """
    
    domain_mappers_swissprot = {}
    domain_mappers_swissprot['mf'] = pickle.load(open('data/processed/domain_mapper_swissprot_mf.p','rb'))
    domain_mappers_swissprot['bp'] = pickle.load(open('data/processed/domain_mapper_swissprot_bp.p','rb'))
    domain_mappers_swissprot['cc'] = pickle.load(open('data/processed/domain_mapper_swissprot_cc.p','rb'))

    
    if (domain not in domain_mappers_swissprot['mf']) and (domain not in domain_mappers_swissprot['bp']) and (domain not in domain_mappers_swissprot['cc']):        
        print(f'Domain {domain} is not present in our SwissProt dataset')
        
    else:        
        go_mappers_swissprot = {}
        go_mappers_swissprot['mf'] = pickle.load(open('data/processed/go_mapper_swissprot_mf.p','rb'))
        go_mappers_swissprot['bp'] = pickle.load(open('data/processed/go_mapper_swissprot_bp.p','rb'))
        go_mappers_swissprot['cc'] = pickle.load(open('data/processed/go_mapper_swissprot_cc.p','rb'))
        
        for onto in ['mf','bp','cc','']:
            if onto=='':                
                print(f'GO term {GO} is not present in our SwissProt dataset')
                
            elif (GO in go_mappers_swissprot[onto]) and (domain in domain_mappers_swissprot[onto]):
                
                mdl_swissprot = DomainGOEmbeddingModel(domain_mappers_swissprot[onto], go_mappers_swissprot[onto])                 # create a model
                mdl_swissprot = load_domaingo_embedding_model_weights(mdl_swissprot, f'saved_models/swissprot_{onto}')   # load model weights
                print(f'Model prediction {round(np.clip(mdl_swissprot(torch.Tensor([domain_mappers_swissprot[onto][domain]]).to(torch.int64),torch.Tensor([go_mappers_swissprot[onto][GO]]).to(torch.int64))[0].detach().numpy()[0][0],0,1),2)}')
                return
                


def main():
    """
    Calculates domain-GO association probability
    """

    domain = ''
    GO = ''

    
    if options.domain:
        domain = options.domain

    if options.GO:
        GO = options.GO



    if len(domain)==0:
        print('Please input a domain')
        sys.exit()

    if len(GO)==0:
        print('Please input a GO term')
        sys.exit()
    
    
    calc_domaingo_prob(domain,GO)


if __name__=='__main__':
    main()