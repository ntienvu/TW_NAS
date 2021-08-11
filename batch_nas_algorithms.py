import sys
sys.path.insert(0,'..')

import pickle
import sys
import copy
import numpy as np
from argparse import Namespace

from data import Data
#from acquisition_functions import acq_fn
#from bo.bo.probo import ProBO
#from bo.dom.list import ListDomain
from bo.pp.pp_gp_my_distmat import MyGpDistmatPP
#from argparse import Namespace

from tqdm import tqdm
from cyDPP.decompose_kernel import decompose_kernel
from cyDPP.sample_dpp import sample_dpp

def run_batch_nas_algorithm(search_space,algo_params):

    # run nas algorithm
    ps = copy.deepcopy(algo_params)
    algo_name = ps['algo_name']
    #algo_name = ps.pop('algo_name')
    
    if algo_name == 'random':
        ps.pop('algo_name')
        ps.pop('batch_size')

        data = random_search(search_space, **ps)
  
    elif "gp" in algo_name:
        data = gp_batch_bayesopt_search(search_space, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()

    k = 1
    if 'k' in ps:
        k = ps['k']

    result_val=compute_best_val_losses(data, k, len(data))
    result_test=compute_best_test_losses(data, k, len(data))
    return result_val,result_test



def compute_best_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i:i[3])[0]
        test_error = best_arch[3]
        results.append((query, test_error))

    return results

def compute_best_val_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i:i[2])[0]
        test_error = best_arch[2]
        results.append((query, test_error))

    return results


def random_search(search_space,
                    total_queries=100, 
                    k=10,
                    allow_isomorphisms=False, 
                    deterministic=True,
                    verbose=1):
    """ 
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries, 
                                                allow_isomorphisms=allow_isomorphisms, 
                                                deterministic_loss=deterministic)
    
    val_losses = [d[2] for d in data]
    
    #top 10
    val_losses = [np.asscalar(d[2]) for d in data]
    top_arches_idx = np.argsort(np.asarray(val_losses))[:10] # descending
    top_arches=[data[ii][0] for ii in top_arches_idx]

    pickle.dump([top_arches,val_losses], open( "10_best_architectures.p", "wb" ) )
        
    print(val_losses[top_arches_idx[0]])
    if verbose:
        top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
        print('Query {}, top 5 val losses {}'.format(total_queries, top_5_loss))   
        
        
        
    return data



def GP_KDPP_Quality(myGP,xtrain,ytrain,xtest,newls,batch_size=5) :
# KDPP for sampling diverse + quality items

    localGP=copy.deepcopy(myGP)
    #data = Namespace()
    #data.X = xtrain
    #data.y = ytrain
    #localGP.set_data(data)
    N=len(xtest)
    mu_test,sig_test=localGP.gp_post(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
    
    score=np.exp(-mu_test)

    qualityK=np.zeros((N,N))+np.eye(N)*score.reshape((-1,1))

    L=qualityK*sig_test*qualityK
	
	# decompose it into eigenvalues and eigenvectors
    vals, vecs = decompose_kernel(L)

    dpp_sample = sample_dpp(vals, vecs, k=batch_size)
    x_t_all=[ xtest[ii] for ii in dpp_sample]
    return x_t_all,dpp_sample

# def GP_KDPP(myGP,xtrain,ytrain,xtest,newls,batch_size=5) :
# # KDPP for sampling diverse + quality items

#     localGP=copy.deepcopy(myGP)
#     mu_test,sig_test=localGP.gp_post(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
    
#     #qualityK=np.zeros((N,N))+np.eye(N)*mu_test.reshape((-1,1))

#     L=sig_test
# 	
# 	# decompose it into eigenvalues and eigenvectors
#     vals, vecs = decompose_kernel(L)

#     dpp_sample = sample_dpp(vals, vecs, k=batch_size)
#     x_t_all=[ xtest[ii] for ii in dpp_sample]
#     return x_t_all,dpp_sample

def optimize_GP_hyper(myGP,xtrain,ytrain,distance):
    # optimizing the GP hyperparameters
    if distance =="tw_distance" or distance=="tw_2_distance" or distance=="tw_2g_distance":
        newls=myGP.optimise_gp_hyperparameter_v3(xtrain,ytrain,alpha=1,sigma=1e-4)
    else:
        newls=myGP.optimise_gp_hyperparameter(xtrain,ytrain,alpha=1,sigma=1e-3)
    return newls
    
def gp_batch_bayesopt_search(search_space,
                        num_init=10,
                        batch_size=5,
                        total_queries=100,
                        distance='edit_distance',
                        algo_name='gp_bucb',
                        deterministic=True,
                        nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        return search_space.query_arch(arch, deterministic=deterministic)[2]

    # this is GP
    modelp = Namespace(kernp=Namespace(ls=0.11, alpha=1, sigma=1e-5), #ls=0.11 for tw
                       infp=Namespace(niter=num_iterations, nwarmup=5),#500
                       distance=distance, search_space=search_space.get_type())
    modelp.distance=distance


	# Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init, 
                                                     deterministic_loss=deterministic)

    xtrain = [d[0] for d in init_data]
    ytrain = np.array([[d[2]] for d in init_data])
    
    # init
    data = Namespace()
    data.X = xtrain
    data.y = ytrain
    
    myGP=MyGpDistmatPP(data,modelp,printFlag=False)

    for ii in tqdm(range(num_iterations)):##
        
        ytrain_scale=(ytrain-np.mean(ytrain))/np.std(ytrain)


        data = Namespace()
        data.X = xtrain
        data.y = ytrain_scale
        
        myGP.set_data(data) #update new data

        xtest=search_space.get_candidate_xtest(xtrain,ytrain)
        xtest=xtest[:100]

         
        
        # this is to enforce to reupdate the K22 between test points
        myGP.K22_d=None
        myGP.K22_d1=None

	    # generate xtest # check here, could be wrong
        #xtest = mylist.unif_rand_sample(500)
        
        if ii%5==0:
            newls=optimize_GP_hyper(myGP,xtrain,ytrain_scale,distance)

        # select a batch of candidate
      
        x_batch,idx_batch=GP_KDPP_Quality(myGP,xtrain,ytrain_scale,xtest,newls,batch_size) 
       
        # evaluate the black-box function
        for xt in x_batch:
            yt=fn(xt)
            xtrain=np.append(xtrain,xt)
            ytrain=np.append(ytrain,yt)
            
        print(np.min(ytrain))
            
    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in xtrain:
        archtuple = search_space.query_arch(arch,deterministic=deterministic)
        results.append(archtuple)

    return results


