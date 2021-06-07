"""
Classes for GP models without any PP backend, using a given distance matrix.
"""

from argparse import Namespace
import time
import copy
import numpy as np
from scipy.spatial.distance import cdist 
from bo.pp.pp_core import DiscPP
from bo.pp.gp.gp_utils import kern_exp_quad, kern_matern32, \
  get_cholesky_decomp, solve_upper_triangular, solve_lower_triangular, \
  sample_mvn, squared_euc_distmat, kern_distmat
from bo.util.print_utils import suppress_stdout_stderr
import scipy
from scipy.optimize import minimize

# this is to make sure that the lengthscale is not overwritten
lengthscale=0.1

class MyGpDistmatPP(DiscPP):
  """ GPs using a kernel specified by a given distance matrix, without any PP
      backend """

  def __init__(self, data=None, modelp=None, printFlag=True):
    """ Constructor """
    self.set_model_params(modelp)
    self.set_data(data)
    self.set_model()
    super(MyGpDistmatPP,self).__init__()
    if printFlag:
      self.print_str()

  def set_model_params(self, modelp):
    """ Set self.modelp """
    if modelp is None:
      pass #TODO
    self.modelp = modelp

  def set_data(self, data):
    """ Set self.data """
    if data is None:
      pass #TODO
      
      Y=np.copy(data.y)
      # normalise the data
      data.y=(Y-np.mean(Y))/np.std(Y)
      #data.X

    self.data_init = copy.deepcopy(data)
    
    self.data = copy.deepcopy(self.data_init)

  def set_model(self):
    """ Set GP regression model """
    self.model = self.get_model()

  def get_model(self):
    """ Returns model object """
    return None


  def infer_post_and_update_samples(self, print_result=True):#False
    """ Update self.sample_list """
    
    global lengthscale

    flagOptimised=False
    if len(self.data.y)%50==0:
        if self.modelp.distance=="tw_3_distance":
            newls=self.optimise_gp_hyperparameter_v3(self.data.X,self.data.y,
                self.modelp.kernp.alpha, self.modelp.kernp.sigma)
        else:
            newls=self.optimise_gp_hyperparameter(self.data.X,self.data.y,
                self.modelp.kernp.alpha, self.modelp.kernp.sigma)
    
        #self.modelp.kernp.ls=np.copy(newls)
        lengthscale=np.copy(newls)
        print(newls, lengthscale)
        flagOptimised=True
        
    
        
    if self.modelp.distance=="tw_3_distance" and np.isscalar(lengthscale) and flagOptimised==False:
        newls=self.optimise_gp_hyperparameter_v3(self.data.X,self.data.y,
                self.modelp.kernp.alpha, self.modelp.kernp.sigma)
        
        lengthscale=np.copy(newls)
        print(newls, lengthscale)
        """
        try:
            old_ls=np.load("lengthscale_tw3_6.npy")
            old_ls=np.vstack((old_ls,lengthscale))
            np.save("lengthscale_tw3_6.npy", old_ls)
        except:
            np.save("lengthscale_tw3_6.npy", lengthscale)
        """
 
    self.modelp.kernp.ls=np.copy(lengthscale)

        
    #self.modelp.kernp.alpha=np.copy(newalpha)
    self.sample_list = [Namespace(ls=self.modelp.kernp.ls,
                                  alpha=self.modelp.kernp.alpha,
                                  sigma=self.modelp.kernp.sigma)]
        
    print(self.modelp.kernp.ls)
    #if print_result: self.print_inference_result()

  def get_distmat(self, xmat1, xmat2):
    """ Get distance matrix """
    #return squared_euc_distmat(xmat1, xmat2, .5)
    
    from data import Data
    
    if "tw" in self.modelp.distance:
        self.distmat = Data.generate_distance_matrix_v3
    else:
        self.distmat = Data.generate_distance_matrix
    return self.distmat(xmat1, xmat2, self.modelp.distance)

  def print_inference_result(self):
    """ Print results of stan inference """
    print('*ls pt est = '+str(self.sample_list[0].ls)+'.')
    print('*alpha pt est = '+str(self.sample_list[0].alpha)+'.')
    print('*sigma pt est = '+str(self.sample_list[0].sigma)+'.')
    print('-----')

  def sample_pp_post_pred(self, nsamp, input_list, full_cov=True):
    """ Sample from posterior predictive of PP.
        Inputs:
          input_list - list of np arrays size=(-1,)
        Returns:
          list (len input_list) of np arrays (size=(nsamp,1))."""
    samp = self.sample_list[0]
    
    try:
        if self.modelp.distance=="tw_3_distance":
            postmu, postcov = self.gp_post_v3(self.data.X, self.data.y, input_list,
                                       samp.ls, samp.alpha, samp.sigma, full_cov)
        else:
            postmu, postcov = self.gp_post(self.data.X, self.data.y, input_list,
                                       samp.ls, samp.alpha, samp.sigma, full_cov)
    except:
        print("bug self.gp_post")
        if self.modelp.distance=="tw_3_distance":
            postmu, postcov = self.gp_post_v3(self.data.X, self.data.y, input_list,
                                       samp.ls, samp.alpha, samp.sigma, full_cov)
        else:
            postmu, postcov = self.gp_post(self.data.X, self.data.y, input_list,
                                       samp.ls, samp.alpha, samp.sigma, full_cov)
    if full_cov:
        try:
            ppred_list = list(sample_mvn(postmu, postcov, nsamp))
        except:
            print("bug ppred_list = list(sample_mvn(postmu, postcov, nsamp))")
            ppred_list = list(sample_mvn(postmu, postcov, nsamp))

    else:
      postcov = np.nan_to_num(postcov)  

      ppred_list = list(np.random.normal(postmu.reshape(-1,),
                                         postcov.reshape(-1,),
                                         size=(nsamp, len(input_list))))
    return list(np.stack(ppred_list).T), ppred_list

  def sample_pp_pred(self, nsamp, input_list, lv=None):
    """ Sample from predictive of PP for parameter lv.
        Returns: list (len input_list) of np arrays (size (nsamp,1))."""
    if lv is None:
      lv = self.sample_list[0]
    postmu, postcov = self.gp_post(self.data.X, self.data.y, input_list, lv.ls,
                                   lv.alpha, lv.sigma)
    pred_list = list(sample_mvn(postmu, postcov, 1)) ###TODO: sample from this mean nsamp times
    return list(np.stack(pred_list).T), pred_list


  def optimise_gp_hyperparameter_v3(self,X,y,alpha,sigma):
    # jointly optimise GP hyperparameter and optimal transport hyper

    opts ={'maxiter':200,'maxfun':200,'disp': True}

    #bounds=np.asarray([[1e-1,5],[0.8,1.5]])
    bounds=np.asarray([[1e-3,0.1],[0.001,0.2],[0.001,0.2],[1e-3,1e-2]])

    self.KK_d1=None
    
    T=4
    best_llk=-np.inf
    best_theta=[0.1]
    for ii in range(T):

        rand_lengthscales=np.random.uniform(bounds[:,0],bounds[:,1],(30,4))
        
        llk=[self.log_llk_v3(X,y,theta=x,alpha=alpha,sigma=sigma) for x in rand_lengthscales]
        #rand_var=np.random.uniform(bounds[1,0],bounds[1,1],1) # randomly select the initial point

        x0=rand_lengthscales[np.argmax(llk)]

        res = minimize(lambda x: -self.log_llk_v3(X,y,theta=x,alpha=alpha,sigma=sigma),x0,
                       bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B

        llk=self.log_llk_v3(X,y,theta=res.x,alpha=alpha,sigma=sigma)

        if llk>=best_llk:
            best_llk,best_theta=llk,res.x

    print(best_theta,best_llk)
    return best_theta

  def optimise_gp_hyperparameter_v2(self,X,y,alpha,sigma):
    # jointly optimise GP hyperparameter and optimal transport hyper

    opts ={'maxiter':200,'maxfun':200,'disp': True}

    #bounds=np.asarray([[1e-1,5],[0.8,1.5]])
    bounds=np.asarray([[1e-2,2],[1e-2,2]])

    self.KK_d1=None
    
    T=3
    best_llk=-np.inf
    best_theta=[0.1]
    for ii in range(T):

        rand_lengthscales=np.random.uniform(bounds[:,0],bounds[:,1],(20,2))
        
        llk=[self.log_llk_v3(X,y,theta=x,alpha=alpha,sigma=sigma) for x in rand_lengthscales]
        #rand_var=np.random.uniform(bounds[1,0],bounds[1,1],1) # randomly select the initial point

        x0=rand_lengthscales[np.argmax(llk)]

        res = minimize(lambda x: -self.log_llk_v3(X,y,theta=x,alpha=alpha,sigma=sigma),x0,
                       bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B

        llk=self.log_llk_v3(X,y,theta=res.x,alpha=alpha,sigma=sigma)

        if llk>best_llk:
            best_llk,best_theta=llk,res.x

    print(best_theta)
    return best_theta



  def optimise_gp_hyperparameter(self,X,y,alpha,sigma):

    opts ={'maxiter':200,'maxfun':200,'disp': True}

    #bounds=np.asarray([[1e-1,5],[0.8,1.5]])
    bounds=np.asarray([[1e-2,3]])

    self.KK_dist=None
    
    T=2
    best_llk=-np.inf
    best_theta=[0.1]
    for ii in range(T):

        rand_lengthscale=np.random.uniform(bounds[0,0],bounds[0,1],30)
        
        llk=[self.log_llk(X,y,theta=x,alpha=alpha,sigma=sigma) for x in rand_lengthscale]
        #rand_var=np.random.uniform(bounds[1,0],bounds[1,1],1) # randomly select the initial point

        x0=rand_lengthscale[np.argmax(llk)]

        res = minimize(lambda x: -self.log_llk(X,y,theta=x,alpha=alpha,sigma=sigma),x0,
                       bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B

        llk=self.log_llk(X,y,theta=res.x,alpha=alpha,sigma=sigma)

        if llk>best_llk:
            best_llk,best_theta=llk,res.x

    #print(best_theta)
    return best_theta


  def log_llk_v3(self,x_train_list, y_train_arr, theta, alpha, sigma):
    """ return the marginal log likelihood"""
    
    #print(theta)
    if len(theta)==4:
        ls1,ls2,ls3, sigma=theta
    else:
        ls1,ls2=theta

    if self.KK_d1 is None:
        self.KK_d1,self.KK_d2,self.KK_d3=self.get_distmat(x_train_list,x_train_list)
        #kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat) #return three values
        if np.any(self.KK_d1)<0:
            print("if np.any(self.KK_dist)<0:")
        
    if len(theta)==3:
        KK=alpha * np.exp(-self.KK_d1/ls1-self.KK_d2/ls2-self.KK_d3/ls3)+np.eye(len(x_train_list))*sigma
    else:
        KK=alpha * np.exp(-self.KK_d1/ls1-(self.KK_d2+self.KK_d3)/ls2)+np.eye(len(x_train_list))*sigma

    try:
        L=scipy.linalg.cholesky(KK,lower=True)
        alpha=np.linalg.solve(KK,y_train_arr)

    except: # singular
        return -np.inf

    try:
        first_term=-0.5*np.dot(y_train_arr.T,alpha)
  
        #chol  = spla.cholesky(KK, lower=True)
        W_logdet=np.sum(np.log(np.diag(L)))
        
        second_term=-W_logdet
        
    except: # singular
        return -np.inf

    logmarginal=first_term+second_term-0.5*len(y_train_arr)*np.log(2*3.14)
    return np.asscalar(logmarginal)



  def log_llk(self,x_train_list, y_train_arr, theta, alpha, sigma):
    """ return the marginal log likelihood"""
    
    ls=theta
    
    if self.KK_dist is None:
        kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
        KK=kernel(x_train_list, x_train_list, ls, alpha)+np.eye(len(x_train_list))*sigma
    
        self.KK_dist=-(ls**2)*np.log((KK-np.eye(len(x_train_list))*sigma)/alpha)
        
        if np.any(self.KK_dist)<0:
            print("if np.any(self.KK_dist)<0:")
   
    KK=alpha * np.exp(-self.KK_dist/(ls**2))+np.eye(len(x_train_list))*sigma
    
    try:
        L=scipy.linalg.cholesky(KK,lower=True)
        alpha=np.linalg.solve(KK,y_train_arr)

    except: # singular
        return -np.inf

    try:
        first_term=-0.5*np.dot(y_train_arr.T,alpha)
        #chol  = spla.cholesky(KK, lower=True)
        W_logdet=np.sum(np.log(np.diag(L)))
        second_term=-W_logdet
        
    except: # singular
        return -np.inf

    logmarginal=first_term+second_term-0.5*len(y_train_arr)*np.log(2*3.14)
    return np.asscalar(logmarginal)

  def gp_post_cache(self, x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma):
    """ Compute parameters of GP posterior """
    # given the caches of k11 and 
    # identify v3 or not
    temp=self.get_distmat(x_train_list[0:2],x_train_list[0:2])
    if len(temp)==3:
        return self.gp_post_cache_v3(x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma)
    
    if not hasattr(self, 'K11_d'):
        self.K11_d=self.get_distmat(x_train_list,x_train_list)
    else: 
        old_N=self.K11_d.shape[0]
        new_N=len(x_train_list)
        if old_N<new_N:# append to [ [_,B],[C,D]]
            B=self.get_distmat(x_train_list[:old_N],x_train_list[old_N:])
            C=B.T
            D=self.get_distmat(x_train_list[old_N:],x_train_list[old_N:])
            
            AB=np.hstack((self.K11_d,B))
            CD=np.hstack((C,D))
            self.K11_d=np.vstack((AB,CD))

    if not hasattr(self, 'K21_d') or self.K21_d is None: # assuming that xtest doesnot change, xtrain has been appended
        self.K21_d=self.get_distmat(x_pred_list,x_train_list)
    else: 
        old_N=self.K21_d.shape[1]
        new_N=len(x_train_list)
        if old_N<new_N:# append to [ [_,B]]
            B=self.get_distmat(x_pred_list,x_train_list[old_N:])
            
            self.K21_d=np.hstack((self.K21_d,B))
            
    if not hasattr(self, 'K22_d') or self.K22_d is None: # xtest doesnot change, dont need to recompute
        self.K22_d=self.get_distmat(x_pred_list,x_pred_list)

    kernel = lambda a,b,c: c*np.exp(-a/b)
    #kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
    k11_nonoise = kernel(self.K11_d, ls, alpha)#+np.eye(len(x_train_list))*sigma
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat,y_train_arr))
    k21 = kernel(self.K21_d, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(self.K22_d, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
   
    return mu2, k2

  def gp_post_cache_v3(self, x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma):
    """ Compute parameters of GP posterior """
    # given the caches of k11 and 
    if not hasattr(self, 'K11_d1'):
        self.K11_d1,self.K11_d2,self.K11_d3=self.get_distmat(x_train_list,x_train_list)
    else: 
        old_N=self.K11_d1.shape[0]
        new_N=len(x_train_list)
        if old_N<new_N:# append to [ [_,B],[C,D]]
            B=self.get_distmat(x_train_list[:old_N],x_train_list[old_N:])
            C=[0]*3
            C[0]=B[0].T
            C[1]=B[1].T
            C[2]=B[2].T
            #B.T
            D=self.get_distmat(x_train_list[old_N:],x_train_list[old_N:])
            
            AB=np.hstack((self.K11_d1,B[0]))
            CD=np.hstack((C[0],D[0]))
            self.K11_d1=np.vstack((AB,CD))
            
            AB=np.hstack((self.K11_d2,B[1]))
            CD=np.hstack((C[1],D[1]))
            self.K11_d2=np.vstack((AB,CD))
            
            AB=np.hstack((self.K11_d3,B[2]))
            CD=np.hstack((C[2],D[2]))
            self.K11_d3=np.vstack((AB,CD))

    if not hasattr(self, 'K21_d1'): # assuming that xtest doesnot change, xtrain has been appended
        self.K21_d1,self.K21_d2,self.K21_d3=self.get_distmat(x_pred_list,x_train_list)
    else: 
        old_N=self.K21_d1.shape[1]
        new_N=len(x_train_list)
        if old_N<new_N:# append to [ [_,B]]
            B=self.get_distmat(x_pred_list,x_train_list[old_N:])
            
            self.K21_d1=np.hstack((self.K21_d1,B[0]))
            self.K21_d2=np.hstack((self.K21_d2,B[1]))
            self.K21_d3=np.hstack((self.K21_d3,B[2]))

    if not hasattr(self, 'K22_d1') or self.K22_d1 is None: # xtest doesnot change, dont need to recompute
        self.K22_d1,self.K22_d2,self.K22_d3=self.get_distmat(x_pred_list,x_pred_list)

    if len(ls)==4:
        sigma=ls[-1]
        kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-b/d[1]-c/d[2])
    if len(ls)==2:
        kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-(b+c)/d[1])
            #kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
        
    k11_nonoise = kernel(self.K11_d1,self.K11_d2,self.K11_d3, ls, alpha)+np.eye(len(x_train_list))*sigma
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat,y_train_arr))
    k21 = kernel(self.K21_d1,self.K21_d2,self.K21_d3, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(self.K22_d1,self.K22_d2,self.K22_d3, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
   
    return mu2, k2

        
        
  def gp_post(self, x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma,
              full_cov=True):
    """ Compute parameters of GP posterior """
    temp=self.get_distmat(x_train_list,x_train_list)
    if len(temp)==3: # tw_3_distance
        self.K11_d1,self.K11_d2,self.K11_d3=temp
        self.K21_d1,self.K21_d2,self.K21_d3=self.get_distmat(x_pred_list,x_train_list)
        self.K22_d1,self.K22_d2,self.K22_d3=self.get_distmat(x_pred_list,x_pred_list)

        if len(ls)==4:
            sigma=ls[-1]
            kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-b/d[1]-c/d[2])
        if len(ls)==2:
            kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-(b+c)/d[1])
        
        self.k11_nonoise = kernel(self.K11_d1,self.K11_d2,self.K11_d3, ls, alpha)+np.eye(len(x_train_list))*sigma
        k21 = kernel(self.K21_d1,self.K21_d2,self.K21_d3, ls, alpha)
        k22 = kernel(self.K22_d1,self.K22_d2,self.K22_d3, ls, alpha)
        
    else:
        self.K11_d=temp
        self.K21_d=self.get_distmat(x_pred_list,x_train_list)
        self.K22_d=self.get_distmat(x_pred_list,x_pred_list)

        kernel = lambda a,b,c: c*np.exp(-a/b)
        #kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
        
        self.k11_nonoise = kernel(self.K11_d, ls, alpha)+np.eye(len(x_train_list))*sigma
        k21 = kernel(self.K21_d, ls, alpha)
        k22 = kernel(self.K22_d, ls, alpha)
        
    lmat = get_cholesky_decomp(self.k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat,y_train_arr))
    vmat = solve_lower_triangular(lmat, k21.T)
    mu2 = k21.dot(smat)

    k2 = k22 - vmat.T.dot(vmat)
    
    if full_cov is False:
      k2_diag=np.diag(k2)
      k2_diag = np.nan_to_num(k2_diag)  
      k2 = np.sqrt(k2_diag)
    return mu2, k2


#  def gp_post_v3(self, x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma,
#              full_cov=True):
#    """ Compute parameters of GP posterior """
#    self.K11_d1,self.K11_d2,self.K11_d3=self.get_distmat(x_train_list,x_train_list)
#    #kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
#    if len(ls)==3:
#        kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-b/d[1]-c/d[2])
#    if len(ls)==2:
#        kernel = lambda a,b,c,d,e: e*np.exp(-a/d[0]-(b+c)/d[1])
#
#    k11_nonoise = kernel(self.K11_d1,self.K11_d2,self.K11_d3, ls, alpha)#+np.eye(len(x_train_list))*sigma
#    self.k11_nonoise=k11_nonoise
#    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
#    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat,
#                                  y_train_arr))
#    
#    self.K21_d1,self.K21_d2,self.K21_d3=self.get_distmat(x_pred_list,x_train_list)
#    k21 = kernel(self.K21_d1,self.K21_d2,self.K21_d3, ls, alpha)
#    mu2 = k21.dot(smat)
#    
#    self.K22_d1,self.K22_d2,self.K22_d3=self.get_distmat(x_pred_list,x_pred_list)
#    k22 = kernel(self.K22_d1,self.K22_d2,self.K22_d3, ls, alpha)
#    vmat = solve_lower_triangular(lmat, k21.T)
#    k2 = k22 - vmat.T.dot(vmat)
#    
#    #print("full_cov",full_cov)
#    if full_cov is False:
#      k2_diag=np.diag(k2)
#      k2_diag = np.nan_to_num(k2_diag)  
#      k2 = np.sqrt(k2_diag)
#  
#    return mu2, k2


  # Utilities
  def print_str(self):
    """ Print a description string """
    print('*MyGpDistmatPP with modelp='+str(self.modelp)+'.')
    print('-----')
