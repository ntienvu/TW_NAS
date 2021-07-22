# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:27:01 2021

@author: Vu Nguyen
"""

from nas_201_api import NASBench201API as API
#api = API('NAS-Bench-201-v1_0-e61699.pth')
api = API('NAS-Bench-201-v1_1-096897.pth')
# Create an API without the verbose log
#api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)
# The default path for benchmark file is '{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')
#api = API(None)

num = len(api)
for i, arch_str in enumerate(api):
  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
  
  
  # show all information for a specific architecture
api.show(1)
api.show(2)

# show the mean loss and accuracy of an architecture
info = api.query_meta_info_by_index(1)  # This is an instance of `ArchResults`
res_metrics = info.get_metrics('cifar10', 'train') # This is a dict with metric names as keys
cost_metrics = info.get_comput_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency

# get the detailed information
results = api.query_by_index(1, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
for seed, result in results.items():
  print ('Latency : {:}'.format(result.get_latency()))
  print ('Train Info : {:}'.format(result.get_train()))
  print ('Valid Info : {:}'.format(result.get_eval('x-valid')))
  print ('Test  Info : {:}'.format(result.get_eval('x-test')))
  # for the metric after a specific epoch
  print ('Train Info [10-th epoch] : {:}'.format(result.get_train(10)))
  
  
  
  
index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
api.show(index)