# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:00:22 2020

@author: Lenovo
"""

import numpy as np

#from numpy.random import permutation
from scipy.sparse.csgraph import shortest_path

def AccumulateLayerOrder(u0, layerInfo):

#    % Input:
#    % u0: degree at each layer
#    % layerInfo: layer order information
#    
#    % Output:
#    % Accumulate degree at each ordered layers
    
    maxLL = int(max(layerInfo))
    u = [0]*(maxLL+1)
    
    for ii in range(maxLL+1):
        u[ii] = np.sum(u0[np.where(layerInfo==ii)])
    
    #print(u)
    return u



def TW_Chain(u,v):# Tree-Wasserstein for a chain
    
    EPS = 1e-10 # for comparing with 0
    
    n,m = len(u),len(v)
    i,j,dd = 0,0,0
    
    while ((i < n) and (j < m)):
        if u[i] <= v[j]:
            dd = dd + u[i]*np.abs((i/n) - (j/m))
            v[j] = v[j] - u[i]
            i+=1;
            if v[j] <= EPS:
                j+=1
        else:
            dd = dd + v[j]*np.abs((i/n) - (j/m))
            u[i] = u[i] - v[j]
            j+=1
            if u[i] <= EPS:
                i+=1
    return dd
    



def TW_InDegrees_NASBENCH(MX, MY,layerX, layerY):
    # sum column
    u0=np.sum(MX,axis=0)
    v0=np.sum(MY,axis=0)
    
#   % Accumulate to vectors using layer_order (also remove isolated nodes)
    u = AccumulateLayerOrder(u0, layerX);
    v = AccumulateLayerOrder(v0, layerY);

    #u=u/np.sum(u)
    #v=v/np.sum(v)
    if np.sum(u)==0:
        u=np.ones(len(u))/len(u)
        #print(u)
    else:
        u=u/np.sum(u)
    if np.sum(v)==0:
        v=np.ones(len(v))/len(v)
        #print(v)
    else:
        v=v/np.sum(v)
        
    #print(layerX,layerY)

    #print(u0,v0)
    #print(u,v)
    dd = TW_Chain(u, v);
    
    #print("dd",dd)
    return dd


def TW_OutDegrees_NASBENCH(MX, MY,layerX, layerY):
    # sum row
    u0=np.sum(MX,axis=1)
    v0=np.sum(MY,axis=1)
    
    #   % Accumulate to vectors using layer_order (also remove isolated nodes)
    u = AccumulateLayerOrder(u0, layerX);
    v = AccumulateLayerOrder(v0, layerY);
    
#    u=u/np.sum(u)
#    v=v/np.sum(v)
    if np.sum(u)==0:
        u=np.ones(len(u))/len(u)
        #print(u)
    else:
        u=u/np.sum(u)
    if np.sum(v)==0:
        v=np.ones(len(v))/len(v)
        #print(v)
    else:
        v=v/np.sum(v)
    dd=TW_Chain(u,v)
    return dd



def TreeMetric_Mapping_2G_NASBENCH101(uu):
    
    vv = [0]*21

    vv[:12] = uu
    # 1-gram subtree
    # -- conv
    vv[12] = uu[0] + uu[1]
    # 1-gram
    vv[13] = np.sum(uu[:2])
    
    # 2-gram subtree
    # ---- s-conv
    vv[14] = uu[3] + uu[7]
    # ---- d-conv
    vv[15] = uu[4] + uu[6]
    # -- conv-conv
    vv[16] = uu[3] + uu[7] + uu[4] + uu[6]
    
    # ---- conv-pool
    vv[17] = uu[5] + uu[8]
    # ---- pool-conv
    vv[18] = uu[9] + uu[10]
    # -- mix-conv&pool
    vv[19] = uu[5] + uu[8] + uu[9] + uu[10]
    
    # 2-gram
    vv[20] = np.sum(uu[3:11])


    return np.asarray(vv)
    
def TreeMetric_Mapping_2G_NASBENCH201(uu):
#    % Input:
#    % uu: 2-gram count vector of operations
#    % 2grams count vector
#    
#    % ---NOTE---
#    % NASBENCH201
#    % for 4 operations
#    % conv1
#    % conv3
#    % avg_pool
#    % sc
#    
#    % 2grams count vector
#    % dim1: #conv1
#    % dim2: #conv3
#    % dim3: #avg_pool
#    % dim4: #sc
#    
#    % dim5: #conv1-conv1
#    % dim6: #conv1-conv3
#    % dim7: #conv1-avg_pool
#    % dim8: #conv1-sc
#    
#    % dim9:  #conv3-conv1
#    % dim10: #conv3-conv3
#    % dim11: #conv3-avg_pool
#    % dim12: #conv3-sc
#    
#    % dim13: #avg_pool-conv1
#    % dim14: #avg_pool-conv3
#    % dim15: #avg_pool-avg-pool
#    % dim16: #avg_pool-sc
#    
#    % dim17: #sc-conv1
#    % dim18: #sc-conv3
#    % dim19: #sc-avg-pool
#    % dim20: #sc-sc
#    
#    % Input
#    % u, v: 2-grams count vector (dim x 1) where dim = 20 for NASBENCH201
#    % e.g., u = [#conv1; #conv3; #avg_pool; #sc; ...
#    %            #conv1-conv1; #conv1-conv3; #conv1-avg_pool; #conv1-sc; ...
#    %            #conv3-conv1; #conv3-conv3; #conv3-avg_pool; #conv3-sc; ...
#    %            #avg_pool-conv1; #avg_pool-conv3; #avg_pool-avg_pool; #avg_pool-sc; ...
#    %            #sc-conv1; #sc-conv3; #sc-avg_pool; #sc-sc]
#    % note: using the above format for count vector

    vv = [0]*33
    
    vv[0:20] = uu;
#    % 1-gram subtree
#    % -- conv
    vv[20] = uu[1] + uu[2];
#    % 1-gram
    vv[21] = sum(uu[:3]);
    
#    % 2-gram subtree
#    % ---- s-conv
    vv[22] = uu[4] + uu[9]
#    % ---- d-conv
    vv[23] = uu[5] + uu[8]
#    % -- conv-conv
    vv[24] = uu[4] + uu[9] + uu[5] + uu[8]
    
#    % ---- conv-pool
    vv[25] = uu[6] + uu[10]
#    % ---- pool-conv
    vv[26] = uu[12] + uu[13]
#    % -- mix-conv&pool
    vv[27] = uu[6] + uu[10] + uu[12] + uu[13]
    
#    % ---- conv-sc
    vv[28] = uu[7] + uu[11]
#    % ---- sc-conv
    vv[29] = uu[16] + uu[17]
#    % -- mix-conv&sc
    vv[30] = uu[7] + uu[11] + uu[16] + uu[17]
    
#    % -- mix-sc&pool
    vv[31] = uu[18] + uu[15]
    
#    % 2-gram
    vv[32] = sum(uu[4:19]);
    

    return np.asarray(vv)


def GetTreeMetric_2Grams_Operations_NASBENCH101(alpha):
    ww = [alpha, alpha, 1, alpha*alpha, alpha*alpha,
     alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha,
     alpha*alpha, 1, (1-alpha), 1, alpha*(1-alpha),
     alpha*(1-alpha), (1-alpha), alpha*(1-alpha), alpha*(1-alpha), (1-alpha),
     1]
    
    return ww
    
def TW_Operations_v2_NASBENCH101(u,v,alpha=0.1):
    # normalisation
    u=u/np.sum(u)
    v=v/np.sum(v)
     
    """
    if np.sum(u) == 0:
        u = np.ones(len(u))/len(u);
    else:
        u = u/np.sum(u);
    
    #vv=vv/np.sum(vv)
    if np.sum(v) == 0:
        v = np.ones(len(v))/len(v);
    else:
        v = v/np.sum(v);
    """
    
    # tree metric mapping
    u_tm = TreeMetric_Mapping_2G_NASBENCH101(u)
    v_tm = TreeMetric_Mapping_2G_NASBENCH101(v)
    
    
    #ww_tm = np.asarray([alpha, alpha, 1, 1-alpha])
    #ww_tm=GetTreeMetric_2Grams_Operations_NASBENCH101(alpha)
    ww_tm = [alpha, alpha, 1, alpha*alpha, alpha*alpha,
     alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha,
     alpha*alpha, 1, (1-alpha), 1, alpha*(1-alpha),
     alpha*(1-alpha), (1-alpha), alpha*(1-alpha), alpha*(1-alpha), (1-alpha),
     1]
    
    # tree-Wasserstein
    dd = np.sum(ww_tm * np.abs(u_tm - v_tm))
    return dd


def TW_Operations_NASBENCH101(u,v,alpha=0.1):
    # normalisation
    u=u/np.sum(u)
    v=v/np.sum(v)
     
    """
    if np.sum(u) == 0:
        u = np.ones(len(u))/len(u);
    else:
        u = u/np.sum(u);
    
    #vv=vv/np.sum(vv)
    if np.sum(v) == 0:
        v = np.ones(len(v))/len(v);
    else:
        v = v/np.sum(v);
    """
    
    # tree metric mapping
    u_tm = TreeMetric_Mapping_2G_NASBENCH101(u)
    v_tm = TreeMetric_Mapping_2G_NASBENCH101(v)
    
    
    #ww_tm = np.asarray([alpha, alpha, 1, 1-alpha])
    #ww_tm=GetTreeMetric_2Grams_Operations_NASBENCH101(alpha)
    ww_tm = [alpha, alpha, 1, alpha*alpha, alpha*alpha,
     alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha, alpha*alpha,
     alpha*alpha, 1, (1-alpha), 1, alpha*(1-alpha),
     alpha*(1-alpha), (1-alpha), alpha*(1-alpha), alpha*(1-alpha), (1-alpha),
     1]
    
    # tree-Wasserstein
    dd = np.sum(ww_tm * np.abs(u_tm - v_tm))
    return dd
    

def TW_Operations_NB101(u,v,alpha=0.1):
    # normalisation
    u=u/np.sum(u)
    v=v/np.sum(v)
    
    # tree metric mapping
    u_tm = u.tolist() +  [u[0] + u[1]]
    v_tm = v.tolist() +[ v[0] + v[1]]
    
    ww_tm = np.asarray([alpha, alpha, 1, 1-alpha])
    
    u_tm=np.asarray(u_tm)
    v_tm=np.asarray(v_tm)
    # tree-Wasserstein
    dd = np.sum(ww_tm * np.abs(u_tm - v_tm))
    return dd

def TW_Operations_NB201(u,v,alpha=0.1):
    # can handle both 1Gram and 2Gram
    
    MaxLength=20 # len for 2gram count vector given 4 operations
    
    if len(u)<MaxLength:
        u=u+[0]*(MaxLength-len(u))
        
    if len(v)<MaxLength:
        v=v+[0]*(MaxLength-len(v))
        
    # normalisation
    u=u/np.sum(u)
    v=v/np.sum(v)
    
    # tree metric mapping
    #    u_tm = u.tolist() +  [u[0] + u[1]]
    #    v_tm = v.tolist() +[ v[0] + v[1]]
    
    u_tm = TreeMetric_Mapping_2G_NASBENCH201(u);
    v_tm = TreeMetric_Mapping_2G_NASBENCH201(v);

    #ww_tm = np.asarray([alpha, alpha, 1, 1-alpha])
    alpha = 0.1;
    a = alpha;
    b = 1-alpha;
    aa = a*a;
    ab = a*b;
    ww_tm  = [a  ,a  ,1  ,1  ,aa , \
              aa ,aa ,aa ,aa ,aa , \
              aa ,aa ,aa ,aa ,1  , \
              a  ,aa ,aa ,a  ,1  , \
              b  ,1  ,ab ,ab ,b  , \
              ab ,ab ,b  ,ab ,ab , \
              b  ,b  ,1]

    u_tm=np.asarray(u_tm)
    v_tm=np.asarray(v_tm)
    # tree-Wasserstein
    dd = np.sum(ww_tm * np.abs(u_tm - v_tm))
    return dd
    


def TW_NASBENCH101(MX, MY, opX, opY, layerX,layerY):
    dd_Operations = TW_Operations_NB101(opX, opY)
    #dd_Degrees = TW_Degrees_NASBENCH(MX, MY)
    dd_InDegrees = TW_InDegrees_NASBENCH(MX, MY,layerX,layerY)
    dd_OutDegrees = TW_OutDegrees_NASBENCH(MX, MY,layerX,layerY)

    #dd=lamb*dd_Operations+(1-lamb)*dd_Degrees
    return dd_Operations,dd_InDegrees,dd_OutDegrees
    
def TW_NASBENCH201(MX, MY, opX, opY, layerX,layerY):
    dd_Operations = TW_Operations_NB201(opX, opY)
    #dd_Degrees = TW_Degrees_NASBENCH(MX, MY)
    dd_InDegrees = TW_InDegrees_NASBENCH(MX, MY,layerX,layerY)
    dd_OutDegrees = TW_OutDegrees_NASBENCH(MX, MY,layerX,layerY)

    #dd=lamb*dd_Operations+(1-lamb)*dd_Degrees
    return dd_Operations,dd_InDegrees,dd_OutDegrees



def TW_2G_NB101(MX, MY, opX, opY, layerX,layerY):
    dd_Operations = TW_Operations_NASBENCH101(opX, opY)
    
    #dd_Degrees = TW_Degrees_v2_NASBENCH101(MX, MY)
    dd_InDegrees = TW_InDegrees_NASBENCH(MX, MY,layerX,layerY)
    dd_OutDegrees = TW_OutDegrees_NASBENCH(MX, MY,layerX,layerY)
    
    #print("dd operation",dd_Operations)
    #print("dd degree",dd_Degrees)
    
    #dd=lamb*dd_Operations+(1-lamb)*dd_Degrees
    
    #dd=(1-lamb1-lamb2)*dd_Operations+lamb1*dd_InDegrees+lamb2*dd_OutDegrees

    #return dd
    return dd_Operations, dd_InDegrees,dd_OutDegrees


def TW_2G_NB201(MX, MY, opX, opY,layerX,layerY):
    dd_Operations = TW_Operations_NB201(opX, opY)
    
    #dd_Degrees = TW_Degrees_v2_NASBENCH101(MX, MY)
    dd_InDegrees = TW_InDegrees_NASBENCH(MX, MY,layerX,layerY)
    dd_OutDegrees = TW_OutDegrees_NASBENCH(MX, MY,layerX,layerY)
    
    #print("dd operation",dd_Operations)
    #print("dd degree",dd_Degrees)
    
    #dd=lamb*dd_Operations+(1-lamb)*dd_Degrees
    
    #dd=(1-lamb1-lamb2)*dd_Operations+lamb1*dd_InDegrees+lamb2*dd_OutDegrees

    #return dd
    return dd_Operations, dd_InDegrees,dd_OutDegrees




MZ = np.asarray([[0, 1, 1, 0, 1, 0, 0],
        [0,0,0,0,1,1, 1],
      [0,0,0,0,0,0, 0],
      [0,0,0,0, 0,0,0],
      [0,0,0,0,0,0, 0],
      [0,0,0,0,0,0, 0],
      [0,0,0,0,0,0,0]])
        
MX = np.asarray([[0, 1, 1, 1, 0, 1, 0],
        [0,0,0,0,0,0, 1],
      [0,0,0,0,0,0, 1],
      [0,0,0,0, 1,0,0],
      [0,0,0,0,0,0, 1],
      [0,0,0,0,0,0, 1],
      [0,0,0,0,0,0,0]])

#% adjacency matrix for network B
MY = np.asarray( [[0, 1, 0, 1, 0, 1, 0],
       [ 1,0,0,1,1,0, 1],
      [0,0,1,0,0,0, 1],
      [0,0,0,0, 1,0,0],
      [0,0,0,0,0,0, 1],
      [0,0,1,0,0,0, 1],
      [0,0,0,0,0,0,0]])

opX=[4,3,2,1,1,2,3,1,4,2,0,1]

opY=[4,3,2,1,1,2,3,1,3,2,0,1]
     

#dd=TW_v2_NASBENCH101(MX,MY,opX,opY)
##print(dd)
#
#dd=TW_v2_NASBENCH101(MZ,MZ,opX,opX)
#print(dd)

layerX=shortest_path(MX,method="D")
layerX[layerX==np.inf]=-1
layerX=layerX[0,:]


layerY=shortest_path(MY,method="D")
layerY[layerY==np.inf]=-1
layerY=layerY[0,:]


dd=TW_2G_NB101(MX,MY,opX,opY,layerX,layerY)
#print(dd)

dd=TW_2G_NB101(MZ,MZ,opX,opX,layerX,layerY)
#print(dd)


dd=TW_2G_NB201(MX,MZ,opX,opX,layerX,layerY)
#print(dd)

