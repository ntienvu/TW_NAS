import sys
sys.path.insert(0,'..')

import itertools
import os
import pickle
import sys
import copy
import numpy as np
import random
import tensorflow as tf
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
    #algo_name = ps['algo_name']
    algo_name = ps.pop('algo_name')

    if algo_name == 'random':
        data = random_search(search_space, **ps)
    elif algo_name == 'evolution':
        data = evolution_search(search_space, **ps)
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


def evolution_search(search_space,
                        num_init=10,
                        k=10,
                        population_size=50,
                        total_queries=100,
                        tournament_size=10,
                        mutation_rate=1.0, 
                        allow_isomorphisms=False,
                        deterministic=True,
                        batch_size=5,
                        verbose=1):
    """
    regularized evolution
    """
    
    data = search_space.generate_random_dataset(num=num_init, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    val_losses = [d[2] for d in data]
    query = num_init

    if num_init <= population_size:
        population = [i for i in range(num_init)]
    else:
        population = np.argsort(val_losses)[:population_size]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = random.sample(population, tournament_size)
        best_index = sorted([(i, val_losses[i]) for i in sample], key=lambda i:i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index][0], mutation_rate)
        #permuted = search_space.perturb_arch(data[best_index][0])

        archtuple = search_space.query_arch(mutated, deterministic=deterministic)
        
        data.append(archtuple)
        val_losses.append(archtuple[2])
        population.append(len(data) - 1)

        # kill the worst from the population
        if len(population) >= population_size:
            worst_index = sorted([(i, val_losses[i]) for i in population], key=lambda i:i[1])[-1][0]
            population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            print('Query {}, top 5 val losses {}'.format(query, top_5_loss))

        query += 1

    return data

def GP_ThompsonSampling(myGP,xtrain,ytrain,xtest,modelp,newls,batch_size=5):

    localGP=copy.deepcopy(myGP)   	# update myGP
    mu_test,sig_test=localGP.gp_post_cache(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
    mu_test=np.ravel(mu_test)
    # normalise mu_test
    idx_all=np.random.choice(len(xtest), batch_size, mu_test)
        
    return xtest[idx_all],idx_all


def GP_BUCB(myGP,xtrain,ytrain,xtest,modelp,newls,batch_size=5):
    # Kriging Believer, Constant Liar for Batch BO
    # minimisation problem

    def LCB(mu,sigma):
        mu=np.reshape(mu,(-1,1))
        sigma=np.reshape(sigma,(-1,1))
        beta_t=2*np.log(100)
        return mu-beta_t*sigma
    
    x_t_all=[0]*batch_size
    idx_all=[]
    
    localGP=copy.deepcopy(myGP)   	# update myGP

    for bb in range(batch_size):
        # update xtrain, ytrain
        data = Namespace()
        data.X = xtrain
        data.y = ytrain
        
        localGP.set_data(data)

        mu_test,sig_test=localGP.gp_post_cache(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
                
        acq_value=LCB(mu_test,np.diag(sig_test))
        idxbest=np.argmin(acq_value )
        
        idx_all=np.append(idx_all,idxbest)
        x_t=xtest[idxbest]
        x_t_all[bb]=x_t
        
        xtrain=np.append(xtrain,x_t)
        ytrain=np.append(ytrain,mu_test[idxbest])
        
    return x_t_all,idx_all


def GP_KDPP_Quality(myGP,xtrain,ytrain,xtest,newls,batch_size=5) :
# KDPP for sampling diverse + quality items

    localGP=copy.deepcopy(myGP)
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

def GP_KDPP(myGP,xtrain,ytrain,xtest,newls,batch_size=5) :
# KDPP for sampling diverse + quality items

    localGP=copy.deepcopy(myGP)
    mu_test,sig_test=localGP.gp_post(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
    
    #qualityK=np.zeros((N,N))+np.eye(N)*mu_test.reshape((-1,1))

    L=sig_test
	
	# decompose it into eigenvalues and eigenvectors
    vals, vecs = decompose_kernel(L)

    dpp_sample = sample_dpp(vals, vecs, k=batch_size)
    x_t_all=[ xtest[ii] for ii in dpp_sample]
    return x_t_all,dpp_sample

def optimize_GP_hyper(myGP,xtrain,ytrain,distance):
    if distance=="tw_3_distance" or distance =="tw_distance":
        newls=myGP.optimise_gp_hyperparameter_v3(xtrain,ytrain,alpha=1,sigma=1e-4)
        #mu_train,sig_train=myGP.gp_post_v3(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-4)
        #mu_test,sig_test=myGP.gp_post_v3(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-4)
    elif distance=="tw_2_distance":
        newls=myGP.optimise_gp_hyperparameter_v2(xtrain,ytrain,alpha=1,sigma=1e-4)
        #mu_train,sig_train=myGP.gp_post_v3(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-4)
        #mu_test,sig_test=myGP.gp_post_v3(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-4)
    else:
        newls=myGP.optimise_gp_hyperparameter(xtrain,ytrain,alpha=1,sigma=1e-3)
        #mu_train,sig_train=myGP.gp_post(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-3)
        #mu_test,sig_test=myGP.gp_post(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
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
        if algo_name=="gp_bucb":
            x_batch,idx_batch=GP_BUCB(myGP,xtrain,ytrain_scale,xtest,modelp,newls,batch_size) 
        elif algo_name=="gp_kdpp":
            x_batch,idx_batch=GP_KDPP(myGP,xtrain,ytrain_scale,xtest,newls,batch_size) 
        elif algo_name=="gp_kdpp_quality":
            x_batch,idx_batch=GP_KDPP_Quality(myGP,ytrain_scale,ytrain,xtest,newls,batch_size) 
        elif algo_name=="gp_kdpp_rand":
            idx_batch = np.random.choice( len(xtest), size=batch_size, replace=False)
            x_batch=[xtest[ii] for ii in idx_batch]
        elif algo_name=="gp_ts":
            x_batch,idx_batch=GP_ThompsonSampling(myGP,xtrain,ytrain_scale,xtest,modelp,newls,batch_size) 
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


