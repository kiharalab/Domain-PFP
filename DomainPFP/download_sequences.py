

import requests
import os
from tqdm import tqdm


def download_sequence(pid):
    """
    Download protein sequence as fasta file

    Args:
        pid (str): UniProt ID of protein
    """

    try:
        os.makedirs('temp_data')
    except:
        pass

    r = requests.get(f'https://rest.uniprot.org/uniprotkb/{pid}.fasta')
    
    if(r.status_code==200):
        raw_fasta = r.text

        fp = open(os.path.join('temp_data',pid+'.fasta'),'w')
        fp.write(raw_fasta)
        fp.close()
        return True

    else:
        print(r.text)
        return False
