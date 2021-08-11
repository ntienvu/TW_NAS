import argparse
import time
import logging
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')

import os
import pickle
import numpy as np

from sequential_nas_algorithms import run_seq_nas_algorithm
#from batch_nas_algorithms import run_batch_nas_algorithm
from params import algo_params_seq,algo_params_batch,meta_neuralnet_params
#from params import *
from data import Data

def run_experiments(args, save_dir):

    from_trials=args.from_trials
    to_trials=args.to_trials
    
    #trials = args.trials
    metann_params = meta_neuralnet_params(args.search_space)
    alg_params = algo_params_seq(args.algo_params)
    num_algos = len(alg_params)
    logging.info(alg_params)
    
    ss = args.search_space
    
    if 'nasbench201' in ss:
        dataset_nb201=ss.split("_", 1)[1]
    else:
        dataset_nb201=None
        
    search_space = Data(ss,dataset_nb201)
    
    results_test = [0]*(to_trials-from_trials)
    results_val = [0]*(to_trials-from_trials)
    
    walltimes = [0]*(to_trials-from_trials)
    
    for j in range(num_algos):
        for i in range(from_trials,to_trials):
            np.random.seed(i)
    
            # run sequential NAS algorithm
            print('\n* Running algorithm: {}'.format(alg_params[j]))
            starttime = time.time()
            algo_result_val,algo_result_test = run_seq_nas_algorithm(search_space,alg_params[j],metann_params)
            algo_result_val = np.round(algo_result_val, 5)
            algo_result_test = np.round(algo_result_test, 5)
    
            # add walltime and results
            walltimes[i-from_trials]=time.time()-starttime
            results_val[i-from_trials]=algo_result_val
            results_test[i-from_trials]=algo_result_test
    
        # print and pickle results
        
        algo_name=alg_params[j]['algo_name']
        if "gp" in alg_params[j]['algo_name']:
            distance=alg_params[j]['distance']
            filename = os.path.join(save_dir, '{}_{}_{}_{}_{}.pkl'.format(from_trials,to_trials,
                                    algo_name,distance,args.search_space))
        else:
            filename = os.path.join(save_dir, '{}_{}_{}_{}.pkl'.format(from_trials,to_trials,
                    algo_name,args.search_space))
        print('\n* Trial summary: (params, results, walltimes)')
        print(alg_params)
       
        print(walltimes)
        print('\n* Saving to file {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([alg_params, results_val,results_test, walltimes], f)
            f.close()
                
            
def main(args):

    # make save directory
    save_dir = args.save_dir
    if not save_dir:
        save_dir = args.algo_params + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for Tree-Wasserstein and k-DPP quality experiments')
    parser.add_argument('--from_trials', type=int, default=0, help='Starting trials index')
    parser.add_argument('--to_trials', type=int, default=5, help='Ending trials index')
    parser.add_argument('--search_space', type=str, default='nasbench201_cifar100', \
                    help='nasbench or nasbench201_cifar10 or nasbench201_cifar100 or nasbench201_ImageNet16-120')
    parser.add_argument('--algo_params', type=str, default='main_experiments', help='which parameters to use')
    #parser.add_argument('--output_filename', type=str, default='cifar100_200iters', help='name of output files')
    parser.add_argument('--save_dir', type=str, default=None, help='name of save directory')
   
    args = parser.parse_args()
    main(args)
