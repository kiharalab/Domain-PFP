from argparse import ArgumentParser
from datetime import datetime
import pickle
import sys, os
sys.path.append(os.path.abspath('DomainPFP'))
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

DOUBLE_PRECISION = 12       # precision for floating point

from domaingo_embedding_model import DomainGOEmbeddingModel, load_domaingo_embedding_model_weights
from domain_embedding import DomainEmbedding
from download_sequences import download_sequence

parser = ArgumentParser()

parser.add_argument('--protein',  help='Uniprot ID of protein', type=str)
parser.add_argument('--fasta',  help='Fasta file of protein sequence', type=str)
parser.add_argument('--savefile', default='emb.p', help='Path to save the protein embeddings (as pickle file)', type=str, required=True )

args = parser.parse_args()


def parse_domains(tsv_file_pth):
    """
    Parses the domains from the InterPro Scan results

    Args:
        tsv_file_pth (str): path to the tsv file
    """

    domains = []
    
    with open(tsv_file_pth,'r') as fp:
        dt = fp.read().split('\n')[:-1]
        for ln in dt:
            domains.append(ln.split('\t')[11])
    
    domains = set(domains)-set({'-'})

    return domains
    

def compute_embeddings(domains):
    """
    Computes the protein embedding from the domains 

    Args:
        domains (set or list): set or list of domains

    Returns:
        tuple of 3 numpy arrays: (MF embedding, BP embedding, CC embedding)
    """
                                                                                    
    domain_mapper_mf = pickle.load(open(os.path.join('data','processed','domain_mapper_swissprot_mf.p'),'rb'))      # loading the mapper files
    go_mapper_mf = pickle.load(open(os.path.join('data','processed','go_mapper_swissprot_mf.p'),'rb'))              
    mdl_mf = DomainGOEmbeddingModel(domain_mapper_mf,go_mapper_mf)                                              # creating a model
    mdl_mf = load_domaingo_embedding_model_weights(mdl_mf, os.path.join('saved_models','swissprot_mf'))             # loading the model weights
    dmn_embedding_mf = DomainEmbedding(mdl_mf, domain_mapper_mf)                                                # creating the Domaing Embedding object

    domain_mapper_bp = pickle.load(open(os.path.join('data','processed','domain_mapper_swissprot_bp.p'),'rb'))
    go_mapper_bp = pickle.load(open(os.path.join('data','processed','go_mapper_swissprot_bp.p'),'rb'))
    mdl_bp = DomainGOEmbeddingModel(domain_mapper_bp,go_mapper_bp)
    mdl_bp = load_domaingo_embedding_model_weights(mdl_bp, os.path.join('saved_models','swissprot_bp'))
    dmn_embedding_bp = DomainEmbedding(mdl_bp, domain_mapper_bp)

    domain_mapper_cc = pickle.load(open(os.path.join('data','processed','domain_mapper_swissprot_cc.p'),'rb'))
    go_mapper_cc = pickle.load(open(os.path.join('data','processed','go_mapper_swissprot_cc.p'),'rb'))
    mdl_cc = DomainGOEmbeddingModel(domain_mapper_cc,go_mapper_cc)
    mdl_cc = load_domaingo_embedding_model_weights(mdl_cc, os.path.join('saved_models','swissprot_cc'))
    dmn_embedding_cc = DomainEmbedding(mdl_cc, domain_mapper_cc)


    mf_embedding = dmn_embedding_mf.get_embedding(-1)           # Initialize embeddings
    bp_embedding = dmn_embedding_bp.get_embedding(-1)
    cc_embedding = dmn_embedding_cc.get_embedding(-1)

    cnt = 0
    for dmn in domains:
        if dmn_embedding_mf.contains(dmn):
            mf_embedding += dmn_embedding_mf.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        mf_embedding /= cnt                     # averaging


    cnt = 0
    for dmn in domains:
        if dmn_embedding_bp.contains(dmn):
            bp_embedding += dmn_embedding_bp.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        bp_embedding /= cnt

    cnt = 0
    for dmn in domains:
        if dmn_embedding_cc.contains(dmn):
            cc_embedding += dmn_embedding_cc.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        cc_embedding /= cnt


    mf_embedding = np.round(mf_embedding,DOUBLE_PRECISION)
    bp_embedding = np.round(bp_embedding,DOUBLE_PRECISION)
    cc_embedding = np.round(cc_embedding,DOUBLE_PRECISION)

    return mf_embedding, bp_embedding, cc_embedding


def main():
    """
    Computes the embeddings of a query protein
    """

    protein = ''
    fasta = ''
    savefile = ''
    
    if args.protein:
        protein = args.protein

    if args.fasta:
        fasta = args.fasta
        
    if args.savefile:
        savefile = args.savefile


    if len(protein)>0:
        print(f'Downloading sequence of {protein} from UniProt')
        flg = download_sequence(protein)
        if(not flg):
            sys.exit()
        else:
            fasta = os.path.join('temp_data',protein+'.fasta')


    elif len(fasta)>0:
        print(f'Loading the fasta file {fasta}')
        if(not os.path.isfile(fasta)):            
            sys.exit(f'{fasta} file not found')

    else:        
        sys.exit('Please input a protein UniProt ID or path to a fasta file')

    if len(savefile)==0:        
        sys.exit('Please input a path to save the protein embeddings')

    try:
        os.makedirs('temp_data',exist_ok=True)
    except Exception as e:
        print(e)
    
    
    job_id = datetime.now().strftime("%H%M%S%f")

    print('Extracting protein domains using InterPro Scan')
    os.system(f'python3 DomainPFP/iprscan5.py --email domainpfp@gmail.com --sequence {fasta} --outfile temp_data/{job_id}')
    print("Domains Computed")
    domains = parse_domains(os.path.join('temp_data',f'{job_id}.tsv.tsv'))

    print("Computing Embeddings")
    mf_embedding, bp_embedding, cc_embedding = compute_embeddings(domains)

    print(f"Saving Embeddings as {savefile}")
    pickle.dump({'mf':mf_embedding,
                 'bp':bp_embedding,
                 'cc':cc_embedding},
                open(savefile,'wb'))


if __name__=='__main__':
    main()