# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:32:50 2020

@author: Lenovo
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


    
Scale_Std=0.5
IsLog=0





   
def Smooth(y,w=5):
    y=list(y)
    y_new=[np.mean(y[max(0,ii-w) :(ii+1)]) for ii in range( len(y))]
    return y_new


def LoadResult(filename,IsLog=IsLog,Scale_Std=Scale_Std,B=5):
    
    temp=pickle.load( open( filename, "rb" ) )
   
    [alg_params, results_val, results_test, walltimes]=temp
        
    # result validation set
    res=np.array(results_val)
    res=res[:,:,1]
    res_mean=np.mean(res[:,:],axis=0)
    res_std_val=Scale_Std*np.std(res,axis=0)
    
    if IsLog==1:
        res_mean=np.log(res_mean)
        res_std_val=Scale_Std*np.std(res_mean)
        
    res_mean=[ np.min(res_mean[:ii+1]) for ii in range(len(res_mean))]
    res_mean_val=Smooth(res_mean)
    
    # result test
    res=np.array(results_test)
    res=res[:,:,1]
    res_mean=np.mean(res[:,:],axis=0)
    res_std_test=Scale_Std*np.std(res,axis=0)
    
    if IsLog==1:
        res_mean=np.log(res_mean)
        res_std_test=Scale_Std*np.std(res_mean)
        
    res_mean=[ np.min(res_mean[:ii+1]) for ii in range(len(res_mean))]
    res_mean_test=Smooth(res_mean)
        
    
    return res_mean_val[::B],res_std_val[::B],res_mean_test[::B],res_std_test[::B]


search_space_list=['nasbench','nasbench201_ImageNet16-120','nasbench201_cifar10','nasbench201_cifar100']



B=5

for search_space in search_space_list:
    
    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)

    # ===================================================
    method='random'
    filename="main_experiments/0_5_B_{:d}_{:s}_{:s}.pkl".format(B,method,search_space)
    [res_mean_val,res_std_val,res_mean_test,res_std_test]=LoadResult(filename)
    
    plt.errorbar(range(0,len(res_mean_test)),res_mean_test,res_std_test,markevery=2,color='m',marker='s',label=method,alpha=0.5)
    
    
    
    # ===================================================
    method='gp_kdpp_quality_tw_distance'
    filename="main_experiments/0_5_B_{:d}_{:s}_{:s}.pkl".format(B,method,search_space)
    [res_mean_val,res_std_val,res_mean_test,res_std_test]=LoadResult(filename)
    plt.errorbar(range(0,len(res_mean_test)),res_mean_test,res_std_test,color='g',marker='P',label="k-DPP Quality TW",alpha=0.5)
    
    
    # ===================================================
    method='gp_kdpp_quality_tw_2g_distance'
    filename="main_experiments/0_5_B_{:d}_{:s}_{:s}.pkl".format(B,method,search_space)
    [res_mean_val,res_std_val,res_mean_test,res_std_test]=LoadResult(filename)
    plt.errorbar(range(0,len(res_mean_test)),res_mean_test,res_std_test,color='r',marker='s',label="k-DPP Quality TW-2G",alpha=0.5)
    
    
    
    plt.legend(prop={'size': 14},ncol=1)
    plt.ylabel('Test Error',fontsize=16)
    plt.xlabel('Iterations',fontsize=16)
    
        
    plt.title(search_space,fontsize=20)


    if search_space=='nasbench':
        plt.ylim([5.7,6.7])
    
    else: # nasbench201
        if 'cifar100' in search_space:
            plt.ylim([38.5,43])    
        elif 'cifar10' in search_space:
            plt.ylim([10.8,14])    
        elif 'ImageNet16-120' in search_space:
            plt.ylim([62,67])    

    strFile="fig/batch_{:s}.png".format(search_space)
    print(strFile)
    fig.savefig(strFile)


