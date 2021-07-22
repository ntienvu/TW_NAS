import numpy as np
import pickle
import sys
from nasbench import api as nb101_api
from nas_bench.cell import Cell,Cell_NB201

from nas_201_api import NASBench201API as API


mysearchspace='A'

class Data:

    def __init__(self, search_space):
        self.search_space = search_space
        global mysearchspace
        mysearchspace=search_space

        if search_space == 'nasbench':
            try:
                self.nasbench = nb101_api.NASBench('../nasbench_only108.tfrecord')
            except:
                self.nasbench = nb101_api.NASBench('../../nasbench_only108.tfrecord')
        if search_space == 'nasbench201':
            try:
                self.nasbench = API('../NAS-Bench-201-v1_1-096897.pth')
            except:
                self.nasbench = API('NAS-Bench-201-v1_1-096897.pth')#e61699
#            self.nasbench = api.NASBench('../../NAS-Bench-201-v1_0-e61699.pth')
            #self.nasbench = api.NASBench('NAS-Bench-201-v1_0-e61699.pth')
            
        if search_space == 'nasbench_full':
            self.nasbench = nb101_api.NASBench('nasbench.tfrecord')

    def get_type(self):
        return self.search_space

    def query_arch(self, 
                    arch=None, 
                    train=True, 
                    encode_paths=True, 
                    deterministic=True, 
                    epochs=50):

        if 'nasbench' in self.search_space: #nasbench 101 and 201
            if arch is None:
                
                if self.search_space=='nasbench201':
                    arch = Cell_NB201.random_cell(nasbench=self.nasbench)
                else:
                    arch = Cell.random_cell(nasbench=self.nasbench)
            if encode_paths:
                if self.search_space=='nasbench201':
                    encoding = Cell_NB201(**arch).encode_paths()
                else:
                    encoding = Cell(**arch).encode_paths()
            else:
                if self.search_space=='nasbench201':
                    encoding = Cell_NB201(**arch).encode_cell()
                else:
                    encoding = Cell(**arch).encode_cell()

            if train:
                if self.search_space=='nasbench201':
                    val_loss = Cell_NB201(**arch).get_val_loss(self.nasbench)
                else:
                    val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
                    
                if self.search_space=='nasbench201':
                    test_loss = Cell_NB201(**arch).get_test_loss(self.nasbench)
                else:
                    test_loss = Cell(**arch).get_test_loss(self.nasbench)
                return (arch, encoding, val_loss, test_loss)
            else:
                return (arch, encoding)
        
        else:
            if arch is None:
                arch = Arch.random_arch()
            if encode_paths:
                encoding = Arch(arch).encode_paths()
            else:
                encoding = arch
                        
            if train:
                val_loss, test_loss = Arch(arch).query(epochs=epochs)
                return (arch, encoding, val_loss, test_loss)
            else:
                return (arch, encoding)

    def mutate_arch(self, arch, mutation_rate=1.0):
        if 'nasbench' in self.search_space:
            if self.search_space=='nasbench201':
                return Cell_NB201(**arch).mutate(self.nasbench, mutation_rate)
            else:
                return Cell(**arch).mutate(self.nasbench, mutation_rate)
        else:
            return Arch(arch).mutate(int(mutation_rate))
        
    def perturb_arch(self, arch, edits=1.0):
        if 'nasbench' in self.search_space:
            if self.search_space=='nasbench201':
                return Cell_NB201(**arch).perturb(self.nasbench,int(edits))
            else:
                return Cell(**arch).perturb(self.nasbench,int(edits))
        else:
            return Arch(arch).perturb()

    def get_path_indices(self, arch):
        if 'nasbench' in self.search_space:
            if self.search_space=='nasbench201':
                return Cell_NB201(**arch).get_path_indices()

            else:
                return Cell(**arch).get_path_indices()
        else:
            return Arch(arch).get_path_indices()[0]

    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                encode_paths=True, 
                                allow_isomorphisms=False, 
                                deterministic_loss=True,
                                patience_factor=5):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break
            archtuple = self.query_arch(train=train,
                                        encode_paths=encode_paths,
                                        deterministic=deterministic_loss)
            path_indices = self.get_path_indices(archtuple[0])

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)

        return data

    def get_candidates(self, data, 
                        num=100,
                        acq_opt_type='mutation',
                        encode_paths=True, 
                        allow_isomorphisms=False, 
                        patience_factor=5, 
                        deterministic_loss=True,
                        num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        for d in data:
            arch = d[0]
            path_indices = self.get_path_indices(arch)
            dic[path_indices] = 1            

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest validation error
            best_arches = [arch[0] for arch in sorted(data, key=lambda i:i[2])[:num_best_arches * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num):
                    mutated = self.mutate_arch(arch)
                    archtuple = self.query_arch(mutated, 
                                                train=False,
                                                encode_paths=encode_paths)
                    path_indices = self.get_path_indices(mutated)

                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1    
                        candidates.append(archtuple)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                archtuple = self.query_arch(train=False, encode_paths=encode_paths)
                path_indices = self.get_path_indices(archtuple[0])

                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(archtuple)

        return candidates


    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            print(self.get_path_indices(d))
            dic[self.get_path_indices(d)] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_path_indices(candidate) not in dic:
                dic[self.get_path_indices(candidate)] = 1
                unduplicated.append(candidate)
        return unduplicated


    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []

        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))

        return data

    # Method used for gp_bayesopt
    def get_arch_list(self,
                        aux_file_path, 
                        distance=None, 
                        iteridx=0, 
                        num_top_arches=10,
                        max_edits=30, 
                        num_repeats=50,
                        verbose=0):

        if self.search_space != 'nasbench':
            print('get_arch_list only supported for nasbench search space')
            sys.exit()

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        
        val_losses = [np.asscalar(d[1]) for d in base_arch_list]
        top_arches_idx = np.argsort(val_losses)[:num_top_arches]
        top_arches=[base_arch_list[ii][0] for ii in top_arches_idx]
        #top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures    
        dic = {}
        for archtuple in base_arch_list:
            path_indices = Cell(**archtuple[0]).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for _ in range(num_repeats):
                #mutated = search_space.mutate_arch(data[best_index][0], mutation_rate)
                mutation = Cell(**arch).mutate(self.nasbench, mutation_rate=1.0)
                #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                path_indices = Cell(**mutation).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(mutation)
                        
        """                
        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    #mutated = search_space.mutate_arch(data[best_index][0], mutation_rate)
                    mutation = Cell(**arch).mutate(self.nasbench, edits)
                    #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    path_indices = Cell(**mutation).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(mutation)
        """
        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = Cell.random_cell(self.nasbench)
                path_indices = Cell(**arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list
    
    
    def get_candidate_xtest(self,
                        xtrain,ytrain, 
                        distance=None, 
                        iteridx=0, 
                        num_top_arches=10,
                        max_edits=30, 
                        num_repeats=30,
                        verbose=0):

        if 'nasbench' not in self.search_space:
            print('get_arch_list only supported for nasbench search space')
            sys.exit()

        # load the list of architectures chosen by bayesopt so far
        #base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        
        val_losses = np.ravel(ytrain)
        top_arches_idx = np.argsort(val_losses)[:num_top_arches]
        top_arches=[xtrain[ii] for ii in top_arches_idx]
        #top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        
        # perturb the best k architectures    
        dic = {}
        for archtuple in xtrain:
            if self.search_space=='nasbench201':
                path_indices = Cell_NB201(**archtuple).get_path_indices()
            else:
                path_indices = Cell(**archtuple).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for _ in range(num_repeats):
                #mutated = search_space.mutate_arch(data[best_index][0], mutation_rate)
                if self.search_space=='nasbench201':
                    mutation = Cell_NB201(**arch).mutate(self.nasbench, mutation_rate=1.0)
                else:
                    mutation = Cell(**arch).mutate(self.nasbench, mutation_rate=1.0)

                #perturbation = Cell(**arch).perturb(self.nasbench, edits)
                if self.search_space=='nasbench201':
                    path_indices = Cell_NB201(**mutation).get_path_indices()
                else:
                    path_indices = Cell(**mutation).get_path_indices()
                    
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(mutation)
                        
   
        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                
                if self.search_space=='nasbench201':
                    arch = Cell_NB201.random_cell(self.nasbench)
                else:
                    arch = Cell.random_cell(self.nasbench)
                path_indices = Cell(**arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list
   
    # Method used for gp_bayesopt for nasbench
    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        
        if mysearchspace=="nasbench201":
            for i, arch_1 in enumerate(arches_1):
                for j in range(len(arches_2)):
                    arch_2=arches_2[j]
                    if distance == 'edit_distance':
                        matrix[i][j] = Cell_NB201(**arch_1).edit_distance(Cell_NB201(**arch_2))
                    elif distance == 'path_distance':
                        matrix[i][j] = Cell_NB201(**arch_1).path_distance(Cell_NB201(**arch_2))        
                    elif distance == 'nasbot_distance': # neural architecture search BO
                        matrix[i][j] = Cell_NB201(**arch_1).nasbot_distance(Cell_NB201(**arch_2))        
                    elif distance == 'gwot_distance': # Gromov Wasserstein + OT from NASBOT
                        matrix[i][j] = Cell_NB201(**arch_1).gwot_distance(Cell_NB201(**arch_2))   
                    elif distance == 'gw_distance': # Gromov Wasserstein
                        matrix[i][j] = Cell_NB201(**arch_1).gw_distance(Cell_NB201(**arch_2))  
#                    elif distance == 'tw_distance': # Tree Wasserstein
#                        matrix[i][j] = Cell_NB201(**arch_1).tw_distance(Cell_NB201(**arch_2))  
#                    elif distance == 'tw_2g_distance': # Tree Wasserstein
#                        matrix[i][j] = Cell_NB201(**arch_1).tw_2gram_distance(Cell_NB201(**arch_2)) 
                    elif distance == 'ot_distance': # OT EMD
                        matrix[i][j] = Cell_NB201(**arch_1).ot_distance(Cell_NB201(**arch_2)) 
                    #elif distance == 'tw_3_distance': # Tree Wasserstein
                        #matrix[i][j] = Cell(**arch_1).tw_3_distance(Cell(**arch_2)) 
                    else:
                        print('{} is an invalid distance'.format(distance))
                        sys.exit()
        else:
            for i, arch_1 in enumerate(arches_1):
                for j in range(len(arches_2)):
                    arch_2=arches_2[j]
                    if distance == 'edit_distance':
                        matrix[i][j] = Cell(**arch_1).edit_distance(Cell(**arch_2))
                    elif distance == 'path_distance':
                        matrix[i][j] = Cell(**arch_1).path_distance(Cell(**arch_2))        
                    elif distance == 'nasbot_distance': # neural architecture search BO
                        matrix[i][j] = Cell(**arch_1).nasbot_distance(Cell(**arch_2))        
                    elif distance == 'gwot_distance': # Gromov Wasserstein + OT from NASBOT
                        matrix[i][j] = Cell(**arch_1).gwot_distance(Cell(**arch_2))   
                    elif distance == 'gw_distance': # Gromov Wasserstein
                        matrix[i][j] = Cell(**arch_1).gw_distance(Cell(**arch_2))  
#                    elif distance == 'tw_distance': # Tree Wasserstein
#                        matrix[i][j] = Cell(**arch_1).tw_distance(Cell(**arch_2))  
#                    elif distance == 'tw_2g_distance': # Tree Wasserstein
#                        matrix[i][j] = Cell(**arch_1).tw_2gram_distance(Cell(**arch_2)) 
                    elif distance == 'ot_distance': # OT EMD
                        matrix[i][j] = Cell(**arch_1).ot_distance(Cell(**arch_2)) 
                    #elif distance == 'tw_3_distance': # Tree Wasserstein
                        #matrix[i][j] = Cell(**arch_1).tw_3_distance(Cell(**arch_2)) 
                    else:
                        print('{} is an invalid distance'.format(distance))
                        sys.exit()
                    
        return matrix
    
    @classmethod
    def generate_distance_matrix_v3(cls, arches_1, arches_2, distance):
        # we will return three separate terms
        
        matrix1 = np.zeros([len(arches_1), len(arches_2)])
        matrix2 = np.zeros([len(arches_1), len(arches_2)])
        matrix3 = np.zeros([len(arches_1), len(arches_2)])
        
        if mysearchspace=="nasbench201":
             for i, arch_1 in enumerate(arches_1):
                 for j in range(len(arches_2)):
                    arch_2=arches_2[j]
                    if distance=='tw_distance':
                        matrix1[i][j],matrix2[i][j],matrix3[i][j] = Cell_NB201(**arch_1).tw_distance(Cell_NB201(**arch_2)) 

                    elif 'tw' in distance: # Tree Wasserstein
                        matrix1[i][j],matrix2[i][j],matrix3[i][j] = Cell_NB201(**arch_1).tw_2g_distance(Cell_NB201(**arch_2)) 
                    else:
                        print('{} is an invalid distance'.format(distance))
                        sys.exit()
        else:
                
            for i, arch_1 in enumerate(arches_1):
                for j in range(len(arches_2)):
                    arch_2=arches_2[j]
                    if distance=='tw_distance':
                        matrix1[i][j],matrix2[i][j],matrix3[i][j] = Cell(**arch_1).tw_distance(Cell(**arch_2)) 
                    elif 'tw' in distance: # Tree Wasserstein
                        matrix1[i][j],matrix2[i][j],matrix3[i][j] = Cell(**arch_1).tw_2g_distance(Cell(**arch_2)) 
                    else:
                        print('{} is an invalid distance'.format(distance))
                        sys.exit()
                    
        #print("max, min value of matrix")
        #print(np.max(matrix),np.min(matrix))
        return matrix1,matrix2,matrix3
