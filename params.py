import sys

def algo_params_seq(param_str):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'main_experiments':

        #params.append({'algo_name':'random','total_queries':200})
        #params.append({'algo_name': 'evolution', 'num_init':20,'total_queries':200})
        
        #params.append({'algo_name':'gp_bayesopt',  'num_init':20,'total_queries':200,'distance':'tw_distance'})  
        params.append({'algo_name':'gp_bayesopt',  'num_init':20,'total_queries':200,'distance':'tw_2g_distance'})  
        #params.append({'algo_name':'bananas', 'num_init':20, 'total_queries':200})   

    else:
        print('invalid algorithm params')
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def algo_params_batch(param_str):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'main_experiments':
        params.append({'algo_name':'random', 'total_queries':300,'batch_size':5})

        params.append({'algo_name':'gp_kdpp_quality', 'batch_size':5, 'num_init':50,
                      'total_queries':100,'distance':'tw_distance'})   
        params.append({'algo_name':'gp_kdpp_quality', 'batch_size':5, 'num_init':50,
                      'total_queries':100,'distance':'tw_2g_distance'})  
    
    else:
        print('invalid algorithm params')
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params

# this is for running the baseline of Bananas
def meta_neuralnet_params(param_str):

    if param_str == 'nasbench':
        params = {'search_space':'nasbench', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
    elif param_str == 'nasbench_full':
        params = {'search_space':'nasbench', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
    elif 'nasbench201' in param_str:
        params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
#        params = {'search_space':'nasbench201', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
#            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
    elif param_str == 'darts':
        params = {'search_space':'darts', 'loss':'mape', 'num_layers':10, 'layer_width':20, \
            'epochs':10000, 'batch_size':32, 'lr':.00001, 'regularization':0, 'verbose':0}

    else:
        print('invalid meta neural net params')
        sys.exit()

    return params



