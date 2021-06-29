#import itertools
#import os
import pickle
import sys
import copy
import numpy as np
import random
import tensorflow as tf
from argparse import Namespace

from data import Data
from acquisition_functions import acq_fn
from meta_neural_net import MetaNeuralnet
#from bo.bo.probo import ProBO
#from bo.dom.list import ListDomain
from bo.pp.pp_gp_my_distmat import MyGpDistmatPP
#from argparse import Namespace
from tqdm import tqdm

def run_seq_nas_algorithm(search_space,algo_params, metann_params):

    # set up search space
#    ss = mp.pop('search_space')
#    search_space = Data(ss)

    # run nas algorithm
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')

    if algo_name == 'random':
        data = random_search(search_space, **ps)
    elif algo_name == 'evolution':
        data = evolution_search(search_space, **ps)
    elif algo_name == 'bananas':
        mp = copy.deepcopy(metann_params)
        data = bananas(search_space, mp, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_search(search_space, **ps)
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
    
    
    #top 10

#    if search_space=="nasbench201":
#        val_losses = [d[2] for d in data]
#    else:
#        val_losses = [np.asscalar(d[2]) for d in data]


    #top_arches_idx = np.argsort(np.asarray(val_losses))[:10] # descending
    #top_arches=[data[ii][0] for ii in top_arches_idx]

    #pickle.dump([top_arches,val_losses], open( "10_best_architectures.p", "wb" ) )
        
    #print(val_losses[top_arches_idx[0]])
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


def bananas(search_space, metann_params,
            num_init=10, 
            k=10, 
            total_queries=150, 
            num_ensemble=5, 
            acq_opt_type='mutation',
            explore_type='its',
            encode_paths=True,
            allow_isomorphisms=False,
            deterministic=True,
            verbose=1):
    """
    Bayesian optimization with a neural network model
    """

    data = search_space.generate_random_dataset(num=num_init, 
                                                encode_paths=encode_paths, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k

    while query <= total_queries:

        xtrain = np.array([d[1] for d in data])
        ytrain = np.array([d[2] for d in data])

        candidates = search_space.get_candidates(data, 
                                                acq_opt_type=acq_opt_type,
                                                encode_paths=encode_paths, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)

        xcandidates = np.array([c[1] for c in candidates])
        predictions = []

        # train an ensemble of neural networks
        train_error = 0
        for _ in range(num_ensemble):
            meta_neuralnet = MetaNeuralnet()
            
            ps = copy.deepcopy(metann_params)
            _ = ps.pop('search_space')

            #train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)
            train_error += meta_neuralnet.fit(xtrain, ytrain, **ps)

            # predict the validation loss of the candidate architectures
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            # clear the tensorflow graph
            tf.reset_default_graph()

        train_error /= num_ensemble
        if verbose:
            print('Query {}, Meta neural net train error: {}'.format(query, train_error))

        # compute the acquisition function for all the candidate architectures
        sorted_indices = acq_fn(predictions, explore_type)

        # add the k arches with the minimum acquisition function values
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(candidates[i][0],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)

        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            print('Query {}, top 5 val losses {}'.format(query, top_5_loss))

        query += k

    return data


def selecting_next_architecture(mu_test,sig_test):

    def LCB(mu,sigma):
        mu=np.reshape(mu,(-1,1))
        sigma=np.reshape(sigma,(-1,1))
        beta_t=2*np.log(100)
        return mu-beta_t*sigma

    acq_value=LCB(mu_test,np.diag(sig_test))
    idxbest=np.argmin(acq_value )
    
    return idxbest

def optimize_GP_hyper(myGP,xtrain,ytrain,distance):
    if distance=="tw_3_distance" or distance=="tw_distance":
        newls=myGP.optimise_gp_hyperparameter_v3(xtrain,ytrain,alpha=1,sigma=1e-4)
        #mu_train,sig_train=myGP.gp_post_v3(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-4)
        #mu_test,sig_test=myGP.gp_post_v3(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-4)
    elif distance=="tw_2g_distance":
        newls=myGP.optimise_gp_hyperparameter_v2(xtrain,ytrain,alpha=1,sigma=1e-4)
        #mu_train,sig_train=myGP.gp_post_v3(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-4)
        #mu_test,sig_test=myGP.gp_post_v3(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-4)
    else:
        newls=myGP.optimise_gp_hyperparameter(xtrain,ytrain,alpha=1,sigma=1e-3)
        #mu_train,sig_train=myGP.gp_post(xtrain,ytrain,xtrain,ls=newls,alpha=1,sigma=1e-3)
        #mu_test,sig_test=myGP.gp_post(xtrain,ytrain,xtest,ls=newls,alpha=1,sigma=1e-3)
    return newls
    
def gp_bayesopt_search(search_space,
                        num_init=10,
                        k=10,
                        total_queries=100,
                        distance='edit_distance',
                        deterministic=True,
                        tmpdir='./',
                        max_iter=200):
    """
    Bayesian optimization with a GP prior
    """
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        return search_space.query_arch(arch, deterministic=deterministic)[2]

    
    # set all the parameters for the various BayesOpt classes
    modelp = Namespace(kernp=Namespace(ls=0.11, alpha=1, sigma=1e-5), #ls=0.11 for tw
                       infp=Namespace(niter=num_iterations, nwarmup=5),#500
                       distance=distance, search_space=search_space.get_type())
    modelp.distance=distance


	# Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init, 
                                                     deterministic_loss=deterministic)
    
    xtrain = [d[0] for d in init_data]
    ytrain = np.array([[d[2]] for d in init_data])

    data = Namespace()
    data.X = xtrain
    data.y = ytrain
    myGP=MyGpDistmatPP(data,modelp,printFlag=False)


    # run Bayesian Optimization
    for ii in tqdm(range(num_iterations)):
        
        ytrain_scale=(ytrain-np.mean(ytrain))/np.std(ytrain)


        data = Namespace()
        data.X = xtrain
        data.y = ytrain_scale
        myGP.set_data(data)
        xtest=search_space.get_candidate_xtest(xtrain,ytrain_scale,num_top_arches=10,max_edits=30)
        xtest=xtest[:100]


        # this is to enforce to reupdate the K22 between test points
        myGP.K22_d=None
        myGP.K22_d1=None
        
        if ii%50==0:
            newls=optimize_GP_hyper(myGP,xtrain,ytrain_scale,distance)
            
        mu_test,sig_test=myGP.gp_post(xtrain,ytrain_scale,xtest,ls=newls,alpha=1,sigma=1e-4)
    
        idxbest=selecting_next_architecture(mu_test,sig_test)
        xt=xtest[idxbest]
        
        # evaluate the black-box function
        yt=fn(xt)
        xtrain=np.append(xtrain,xt)
        ytrain=np.append(ytrain,yt)

                        
    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in xtrain:
        archtuple = search_space.query_arch(arch,deterministic=deterministic)
        results.append(archtuple)

    return results

