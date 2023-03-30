import numpy as np
from tqdm import tqdm

def compute_domain_go_intersection(all_protein_domains, all_protein_gos):
    """
    Determines the co-occuring domains and the GO terms

    Args:
        all_protein_domains (dict): python dictionary containing domains of the individual proteins
        all_protein_gos (dict): python dictionary containing GO term of the individual proteins

    Returns:
        tuple of dicts: (
                            python dict containing the GO terms when a particular domain is present,
                            python dict containing the domains when a particular GO term is present
                        )
    """

    domain_go_intersection = {}         # python dict containing the GO terms when a particular domain is present
    go_domain_intersection = {}         # python dict containing the domains when a particular GO term is present

    for prtn in tqdm(all_protein_gos):

        for dmn in all_protein_domains[prtn]:
            if dmn not in domain_go_intersection:
                    domain_go_intersection[dmn] = set()

            for go_trm in all_protein_gos[prtn]:
                if go_trm not in go_domain_intersection:
                    go_domain_intersection[go_trm] = set()

                domain_go_intersection[dmn].add(go_trm)
                go_domain_intersection[go_trm].add(dmn)

    return (domain_go_intersection, go_domain_intersection)



def compute_domain_go_score(all_domain_proteins,all_go_proteins,domain_mapper, domain_go_intersection):
    """
    Compute all the p(GO|domain) scores

    Args:
        all_domain_proteins (dict): python dictionary containing proteins containing individual domains
        all_go_proteins (dict): python dictionary containing proteins with individual GO terms
        domain_mapper (dict): python dictionary mapping domains to their ids
        domain_go_intersection (dict): python dictionary containing the GO terms when a particular domain is present,

    Returns:
        dict: python dictionary containing all the p(GO|domain) scores
    """


    domain_go_scores = {}                   # initialization


    for domain in domain_mapper:             # iterate over all the domains

        domain_go_scores[domain] = {}

        for go in domain_go_intersection[domain]:

            intersect = all_domain_proteins[domain].intersection(all_go_proteins[go])       # compute p(domain \cap GO)
            
            y = len(intersect)/len(all_domain_proteins[domain])                             # compute p(GO|domain) = p(domain \cap GO) / p(domain)

            domain_go_scores[domain][go] = y
            

    return domain_go_scores




def iprdict():
    fp = open('interpro2go.txt','r')
    dt = fp.read().split('\n')[:-1]
    fp.close()

    ipr2go_pairs = []
    ipr2go_dict = {}

    for ln in tqdm(dt):

        if ln[0]=='!':
            continue

        intrpro_trm, go_trm = ln.split(' ')[0].split(':')[1],ln.split(' ')[-1]

        ipr2go_pairs.append((intrpro_trm, go_trm))

        if intrpro_trm not in ipr2go_dict:
            ipr2go_dict[intrpro_trm] = {}
        ipr2go_dict[intrpro_trm][go_trm] = 1
    return ipr2go_dict

def prepare_embedding_model_data_random(all_domain_proteins,all_go_proteins,all_domains,all_gos,domain_mapper,go_mapper, count = 10000):

    np.random.seed(2)

    ipr2go_dict = iprdict()

    if(len(go_mapper)<count):
        count = len(go_mapper)

    X_domain_idx = np.zeros(len(domain_mapper)*count)
    X_go_idx = np.zeros(len(domain_mapper)*count)
    Y_domaingo = np.zeros(len(domain_mapper)*count)


    i = 0


    for domain_id in tqdm(range(len(domain_mapper))):
        for go_id in np.random.randint(0,len(go_mapper),count):

            domain = all_domains[domain_id]
            go = all_gos[go_id]    

            intersect = all_domain_proteins[domain].intersection(all_go_proteins[go])

            x_domain = domain_id
            x_go = go_id

            y = len(intersect)/len(all_domain_proteins[domain])


            X_domain_idx[i] = x_domain
            X_go_idx[i] = x_go
            Y_domaingo[i] = y

            i += 1

    return (X_domain_idx, X_go_idx, Y_domaingo)


def prepare_embedding_model_data_negative_sampling(all_domain_proteins,all_go_proteins,all_domains,all_gos,domain_mapper,go_mapper, domain_go_intersection, negative_samples_factor = 1):

    np.random.seed(2)

    ipr2go_dict = iprdict()
    ignored = 0

    count = 0

    for dmn in domain_go_intersection:
        count += len(domain_go_intersection[dmn])*(1+negative_samples_factor)

    X_domain_idx = np.zeros(count)
    X_go_idx = np.zeros(count)
    Y_domaingo = np.zeros(count)

    go_set = set(all_gos)

    
    i = 0


    for domain_id in tqdm(range(len(domain_mapper))):

        domain = all_domains[domain_id]

        for go in domain_go_intersection[domain]:

            intersect = all_domain_proteins[domain].intersection(all_go_proteins[go])

            x_domain = domain_id
            x_go = go_mapper[go]

            if True:
                if (domain in ipr2go_dict) and (go in ipr2go_dict[domain]):
                    ignored += 1
                    continue

            y = len(intersect)/len(all_domain_proteins[domain])


            X_domain_idx[i] = x_domain
            X_go_idx[i] = x_go
            Y_domaingo[i] = y

            i += 1

        negative_samples = list(go_set - domain_go_intersection[domain])
        if((len(domain_go_intersection[domain])*negative_samples_factor)>=len(negative_samples)):
            replace = True 
        else:
            replace = False
        
        for go in np.random.choice(negative_samples,len(domain_go_intersection[domain])*negative_samples_factor,replace=replace):

            x_domain = domain_id
            x_go = go_mapper[go]

            y = 0

            X_domain_idx[i] = x_domain
            X_go_idx[i] = x_go
            Y_domaingo[i] = y

            i += 1
    

    print(f'ignored {ignored} pairs')

    return (X_domain_idx, X_go_idx, Y_domaingo)


def prepare_embedding_model_data_balaced(all_domain_proteins,all_go_proteins,all_domains,all_gos,domain_mapper,go_mapper, domain_go_intersection, extra_count=500):    


    np.random.seed(2)

    max_cnt = 100000000

    for dmn in domain_go_intersection:
        max_cnt = max(len(domain_go_intersection[dmn]),max_cnt)

    count = (max_cnt+extra_count)*2*len(domain_go_intersection)

    X_domain_idx = np.zeros(count)
    X_go_idx = np.zeros(count)
    Y_domaingo = np.zeros(count)

    go_set = set(all_gos)

    
    i = 0


    for domain_id in tqdm(range(len(domain_mapper))):

        domain = all_domains[domain_id]

        for go in np.random.choice(domain_go_intersection[domain],max_cnt+extra_count,replace=True):    

            intersect = all_domain_proteins[domain].intersection(all_go_proteins[go])

            x_domain = domain_id
            x_go = go_mapper[go]

            y = len(intersect)/len(all_domain_proteins[domain])


            X_domain_idx[i] = x_domain
            X_go_idx[i] = x_go
            Y_domaingo[i] = y

            i += 1

        negative_samples = list(go_set - domain_go_intersection[domain])

        for go in np.random.choice(negative_samples,max_cnt+extra_count,replace=True):

            x_domain = domain_id
            x_go = go_mapper[go]

            y = 0

            X_domain_idx[i] = x_domain
            X_go_idx[i] = x_go
            Y_domaingo[i] = y

            i += 1
 

    return (X_domain_idx, X_go_idx, Y_domaingo)

