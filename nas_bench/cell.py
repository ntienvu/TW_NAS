import numpy as np
import copy
import itertools
import random
import ot
from nasbench import api
import time
from scipy.sparse.csgraph import shortest_path

from tw_2g_v2b import TW_InDegrees_NASBENCH,TW_OutDegrees_NASBENCH,TW_2G_NB201
from tw_2g_v2b import TW_2G_NB101,TW_Operations_NB101,TW_NASBENCH201,TW_NASBENCH101



class Cell:

    def __init__(self, matrix, ops):

        self.matrix = matrix
        self.ops = ops
        self.get_infor()


    def get_infor(self):
        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.CONV3X3 = 'conv3x3-bn-relu'
        self.CONV1X1 = 'conv1x1-bn-relu'
        self.MAXPOOL3X3 = 'maxpool3x3'
        self.OPS = [self.CONV3X3, self.CONV1X1, self.MAXPOOL3X3]
        self.OPS_2Gram=[]
        self.NUM_VERTICES = 7
        self.OP_SPOTS = self.NUM_VERTICES - 2
        self.MAX_EDGES = 9
        
    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """ 
        From the NASBench repository 
        https://github.com/google-research/nasbench
        """
        
        INPUT = 'input'
        OUTPUT = 'output'
        CONV3X3 = 'conv3x3-bn-relu'
        CONV1X1 = 'conv1x1-bn-relu'
        MAXPOOL3X3 = 'maxpool3x3'
        OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
        OPS_2Gram=[]
        NUM_VERTICES = 7
        OP_SPOTS = NUM_VERTICES - 2
        MAX_EDGES = 9

        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }

    def get_val_loss(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']))
        else:        
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 3)            


    def get_test_loss(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100*(1-np.mean(accs)), 3)

    def perturb(self, nasbench, edits=1):
        """ 
        create new perturbed cell 
        inspird by https://github.com/google-research/nasbench
        """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, self.NUM_VERTICES - 1):
                        for dst in range(src+1, self.NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, self.NUM_VERTICES - 1):
                        available = [op for op in self.OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def mutate(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / self.NUM_VERTICES
            for src in range(0, self.NUM_VERTICES - 1):
                for dst in range(src + 1, self.NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / self.OP_SPOTS
            for ind in range(1, self.OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in self.OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }

    def encode_cell(self):
        """ 
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding 
        """
        encoding_length = (self.NUM_VERTICES ** 2 - self.NUM_VERTICES) // 2 + self.OP_SPOTS
        encoding = np.zeros((encoding_length))
        dic = {self.CONV1X1: 0., self.CONV3X3: 0.5, self.MAXPOOL3X3: 1.0}
        n = 0
        for i in range(self.NUM_VERTICES - 1):
            for j in range(i+1, self.NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, self.NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return tuple(encoding)

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, self.NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, self.NUM_VERTICES - 1):
            for j in range(1, self.NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {self.CONV3X3: 0, self.CONV1X1: 1, self.MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(self.NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(self.OPS) ** i * (mapping[path[i]] + 1)

        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(self.OPS) ** i for i in range(self.OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        """ 
        compute the distance between two architectures
        by comparing their path encodings
        """
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))


    
    def ot_distance(self, other):
        # distance based on OTMANN distance adapted to cell-based search spaces
        # see our arxiv paper for more details
        MAXVAL = 10000;


        MX=self.matrix
        MY=other.matrix
        
        opX=self.get_1gram_count_vector(MX,self.ops,self.OPS)
        opY=self.get_1gram_count_vector(MY,other.ops,self.OPS)
        
        Mcost = np.asarray([[0,0.2,MAXVAL],[0.2,0,MAXVAL],[MAXVAL,MAXVAL,0]])
        # from Table 1 in https://arxiv.org/pdf/1802.07191.pdf
        
        Wd=ot.emd2(opX,opY,Mcost)
        
        return Wd
    
    
    def gwot_distance(self, other):
        # distance based on OTMANN distance adapted to cell-based search spaces
        # see our arxiv paper for more details

        row_sums = sorted(np.array(self.matrix).sum(axis=0))
        col_sums = sorted(np.array(self.matrix).sum(axis=1))

        other_row_sums = sorted(np.array(other.matrix).sum(axis=0))
        other_col_sums = sorted(np.array(other.matrix).sum(axis=1))

        row_dist = np.sum(np.abs(np.subtract(row_sums, other_row_sums)))
        col_dist = np.sum(np.abs(np.subtract(col_sums, other_col_sums)))

        counts = [self.ops.count(op) for op in self.OPS]
        other_counts = [other.ops.count(op) for op in self.OPS]

        ops_dist = np.sum(np.abs(np.subtract(counts, other_counts)))

        n=self.matrix.shape[0]
        p = ot.unif(n)
        q = ot.unif(n)
        C1=self.matrix
        C2=other.matrix
        C1=C1+1e-8
        C2=C2+1e-8
        
        C1 /= C1.max()
        C2 /= C2.max()

        #start = time.time()

        #gw, log = ot.gromov.entropic_gromov_wasserstein(
                #C1, C2, p, q, 'kl_loss', epsilon=1e-3, log=True, verbose=False)

        gw, log = ot.gromov.gromov_wasserstein(
                C1, C2, p, q, 'square_loss', log=True, verbose=False)
        
        dist1=(row_dist + col_dist + ops_dist)/(np.sum(self.matrix)+np.sum(other.matrix))
                
        #end = time.time()
        #print(end - start)

        dist2=(log['gw_dist']-0.05)/0.4

        return dist1+dist2
    
    def gw_distance(self, other):
       # George Andrew D Briggs
       # 0.48 - 0.08

        n=self.matrix.shape[0]
        p = ot.unif(n)
        q = ot.unif(n)
        C1=self.matrix
        C2=other.matrix
        C1=C1+1e-8
        C2=C2+1e-8
        
        C1 /= C1.max()
        C2 /= C2.max()


        #gw, log = ot.gromov.entropic_gromov_wasserstein(
                #C1, C2, p, q, 'square_loss', epsilon=1e-3, log=True, verbose=False)

        gw, log = ot.gromov.gromov_wasserstein(
                C1, C2, p, q, 'square_loss', log=True, verbose=False)
                

        #dist=(log['gw_dist']-0.05)/0.4
        dist=(log['gw_dist'])
        return dist
    
    
    def get_1gram_count_vector(self,MX,ops,OPS):
        tempX=np.sum(MX,axis=1)
        idxRow= set(np.argwhere(tempX).ravel())
        #idxRow=set(np.argwhere(tempX==0))
        
        countX=np.sum(MX,axis=0)
        idxCol= set(np.argwhere(countX).ravel())
        
        idx= list(idxRow.union(idxCol))

        myops=[ops[ii] for ii in idx]
        opX = [myops.count(op) for op in OPS]
        
        return opX

        
    def tw_distance(self, other,lamb=0.5):
        
        MX=self.matrix
        MY=other.matrix
        #Xops=self.ops[1:-1]
        #Yops=other.ops[1:-1]
        
        #MX=MX[1:-1,1:-1] # crop 7x7 to 5x5
        #MY=MY[1:-1,1:-1] # crop 7x7 to 5x5
        
        # remove empty row and empty col
        opX=self.get_1gram_count_vector(MX,self.ops,self.OPS)
        opY=self.get_1gram_count_vector(MY,other.ops,self.OPS)
    
        # get layer order using shortest path
        layerX=shortest_path(MX,method="D")
        layerX[layerX==np.inf]=-1
        layerX=layerX[0,:]
        #layerXOut=layerX[:,0]

        layerY=shortest_path(MY,method="D")
        layerY[layerY==np.inf]=-1
        layerY=layerY[0,:]
        #layerYOut=layerY[:,0]        #opX = [self.ops.count(op) for op in OPS]
        #opY = [other.ops.count(op) for op in OPS]
     
        return TW_NASBENCH101(MX,MY,opX,opY,layerX,layerY)
    
    
    def mapping_operation(self,opsrow,opscol,OPS):
        
        if opsrow==OPS[0]:# cov3x3
            uu=1
        if opsrow==OPS[1]:# cov 1x1
            uu=2
        if opsrow==OPS[2]:# max pooling
            uu=3
            
        if opscol==OPS[0]:# cov3x3
            index=3*uu
            return index
        if opscol==OPS[1]:#cov 1x1
            index=3*uu+1
            return index
        if opscol==OPS[2]:#max pooling
            index=3*uu+2
            return index

        return -1
        
    def count_operation_2gram(self,MX,ops):
        count=[0]*12
        
        # first three dimension 1gram
        count[:3]=self.get_1gram_count_vector(MX,ops,self.OPS)
        
        MX=MX[1:-1,1:-1] # crop 7x7 to 5x5
        ops=ops[1:-1] # remove INPUT, OUTPUT
        
        # process 9 remaining dimension
        for ii in range(MX.shape[0]): # each row
            for jj in range(MX.shape[1]): # each column
                if MX[ii,jj]>0:
                    index=self.mapping_operation(ops[ii],ops[jj],self.OPS)
                    count[index]+=1
                
        return count

#    def tw_2gram_distance(self,other,lamb=0.5):
#        MX=self.matrix
#        MY=other.matrix
#        
#        # remove empty row
#        #tempX=np.sum(MX,axis=1)
#        #idx= np.argwhere(tempX).ravel()
#        #opX=[self.ops[ii] for ii in idx]
#
#        #tempY=np.sum(MY,axis=1)
#        #idx= np.argwhere(tempY).ravel()
#        #opY=[other.ops[ii] for ii in idx]
#        
#        
#        opX = self.count_operation_2gram(MX,self.ops)
#        opY = self.count_operation_2gram(MY,other.ops)
#        #print(opX,opY)
#        dd=TW_v2_NB101(MX,MY,opX,opY)
#    
#        #print(dd)
#        return dd
        
    def tw_2g_distance(self,other):
        MX=self.matrix
        MY=other.matrix
        
        # remove empty row
        #tempX=np.sum(MX,axis=1)
        #idx= np.argwhere(tempX).ravel()
        #opX=[self.ops[ii] for ii in idx]

        #tempY=np.sum(MY,axis=1)
        #idx= np.argwhere(tempY).ravel()
        #opY=[other.ops[ii] for ii in idx]
        
        
        opX = self.count_operation_2gram(MX,self.ops)
        opY = self.count_operation_2gram(MY,other.ops)
        
        layerX=shortest_path(MX,method="D")
        layerX[layerX==np.inf]=-1
        layerX=layerX[0,:]
        #layerXOut=layerX[:,0]

        layerY=shortest_path(MY,method="D")
        layerY[layerY==np.inf]=-1
        layerY=layerY[0,:]
        
        #print(opX,opY)    
        #print(dd)
        return TW_2G_NB101(MX,MY,opX,opY,layerX,layerY) # return 3 elements
         


class Cell_NB201(Cell):

    def __init__(self, matrix, ops):
        #self.dataset='cifar100'
        #self.dataset='ImageNet16-120'
        self.dataset='cifar10'
        self.matrix = matrix
        self.ops = ops
        self.matrix = matrix
        self.ops = ops
        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.CONV3X3 = 'nor_conv_3x3'
        self.CONV1X1 = 'nor_conv_1x1'
        self.AVEPOOL3X3='avg_pool_3x3'
        self.SKIPCONNECT='skip_connect'
        self.NONE='none'
        self.OPS = [self.CONV3X3, self.CONV1X1, self.AVEPOOL3X3,self.SKIPCONNECT,self.NONE]       
        self.OPS_TW = [self.CONV3X3, self.CONV1X1, self.AVEPOOL3X3,self.SKIPCONNECT]

        #self.OPS = [self.CONV3X3, self.CONV1X1, self.AVEPOOL3X3,self.SKIPCONNECT]

        self.OPS_2Gram=[]
        self.NUM_VERTICES = 8
        self.OP_SPOTS = self.NUM_VERTICES - 2
        self.MAX_EDGES = 10


    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def modelspec(self):
        print("not implemented")
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    def Nas201_String_To_OpsMatrix(self,mystr):
    
        tokenCell = mystr.split("+")
        nOperation=8
        listOperation=[0]*int(nOperation)
        #listOperation = cell(length(tokenCell)*(length(tokenCell)+1)/2, 1);
        curID = 0
        listOperation[0] = 'input';
        for ii in range(len(tokenCell)):
            tmpCell = tokenCell[ii].split('|')
            strimTmpCell = tmpCell[1:-1]
            
            for jj in range(len(strimTmpCell)):
                opTmpCell = strimTmpCell[jj].split('~')
                curID = curID + 1;
                listOperation[curID] = opTmpCell[0];
         
        curID = curID + 1
        listOperation[curID] = 'output'
        
        adjacencyMatrix = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0], 
                           [1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 0]])
        
        adjacencyMatrix=adjacencyMatrix.T
        return listOperation,adjacencyMatrix

    def Nas201_OpsMatrix_To_String(self,listOperation,Mat):
        ss = "|" + listOperation[1] + "~0|";
        ss = ss + "+|" + listOperation[2] + "~0|" + listOperation[3] + "~1|";
        ss = ss + "+|" + listOperation[4] + "~0|" + listOperation[5] + "~1|" + listOperation[6] + "~2|";
        return ss

    def is_valid(self):
        return 1

    @classmethod
    def random_cell(cls, nasbench):
        """ 
        From the NASBench repository 
        https://github.com/google-research/nasbench
        """
        
        INPUT = 'input'
        OUTPUT = 'output'
        CONV3X3 = 'nor_conv_3x3'
        CONV1X1 = 'nor_conv_1x1'
        AVEPOOL3X3='avg_pool_3x3'
        SKIPCONNECT='skip_connect'
        NONE='none'
        OPS = [CONV3X3, CONV1X1, AVEPOOL3X3,SKIPCONNECT,NONE]
        OPS_2Gram=[]
        NUM_VERTICES = 8
        OP_SPOTS = NUM_VERTICES - 2
        MAX_EDGES = 10
        
        matrix = np.random.choice(
            [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        
        return { 'matrix': matrix,
                    'ops': ops}


    def get_val_loss(self, nasbench):
        # output one of the three validation accuracies at random
        #return (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']))

        # get index based on the matrix and operation
        ss_query=self.Nas201_OpsMatrix_To_String(self.ops,self.matrix)
        index = nasbench.query_index_by_arch(ss_query)
        #mystr="|avg_pool_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|none~0|nor_conv_3x3~1|none~2|"
        #index = nasbench.query_index_by_arch(mystr)

        results = nasbench.query_by_index(index, self.dataset) # a dict of all trials for 1st net on cifar100, where the key is the seed
        results=results[888]
        
        try:
            accuracy=np.round(results.get_eval('x-valid')['accuracy'],decimals=4)
        except:
            #print(ss_query)
            accuracy=np.round(results.get_eval('ori-test')['accuracy'],decimals=4)


        return 100-accuracy

    def get_test_loss(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        ss_query=self.Nas201_OpsMatrix_To_String(self.ops,self.matrix)
        index = nasbench.query_index_by_arch(ss_query)
        results = nasbench.query_by_index(index, self.dataset) # a dict of all trials for 1st net on cifar100, where the key is the seed
        results=results[888]

        try:
            accuracy=np.round(results.get_eval('x-test')['accuracy'],decimals=4)
        except:
            #print(ss_query)
            accuracy=np.round(results.get_eval('ori-test')['accuracy'],decimals=4)


        return 100-accuracy

    def perturb(self, nasbench, edits=1):
        """ 
        create new perturbed cell 
        inspird by https://github.com/google-research/nasbench
        """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, self.NUM_VERTICES - 1):
                        for dst in range(src+1, self.NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, self.NUM_VERTICES - 1):
                        available = [op for op in self.OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def mutate(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / self.NUM_VERTICES
            for src in range(0, self.NUM_VERTICES - 1):
                for dst in range(src + 1, self.NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / self.OP_SPOTS
            for ind in range(1, self.OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in self.OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)
           
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

    def encode_cell(self):
        """ 
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding 
        """
        encoding_length = (self.NUM_VERTICES ** 2 - self.NUM_VERTICES) // 2 + self.OP_SPOTS
        encoding = np.zeros((encoding_length))
        dic = {self.CONV3X3: 0, self.CONV1X1: 1, self.AVEPOOL3X3: 2, self.SKIPCONNECT:3, self.NONE:4}
        n = 0
        for i in range(self.NUM_VERTICES - 1):
            for j in range(i+1, self.NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, self.NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return tuple(encoding)

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, self.NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, self.NUM_VERTICES - 1):
            for j in range(1, self.NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {self.CONV3X3: 0, self.CONV1X1: 1, self.AVEPOOL3X3: 2, self.SKIPCONNECT:3,self.NONE:4}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(self.NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(self.OPS) ** i * (mapping[path[i]] + 1)

        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(self.OPS) ** i for i in range(self.OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        path_encoding = np.zeros(num_paths)
        
        try:
            for index in path_indices:
                path_encoding[index] = 1
        except:
            print("bug")
            for index in path_indices:
                path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        """ 
        compute the distance between two architectures
        by comparing their path encodings
        """
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def edit_distance(self, other):
        return super(Cell_NB201, self).edit_distance(other)

    def nasbot_distance(self, other):
        return super(Cell_NB201, self).nasbot_distance(other)

    def ot_distance(self, other):
        # distance based on OTMANN distance adapted to cell-based search spaces
        # see our arxiv paper for more details
        MAXVAL = 10000;

        MX=self.matrix
        MY=other.matrix
        
        opX=super(Cell_NB201, self).get_1gram_count_vector(MX,self.ops,self.OPS_TW)
        opY=super(Cell_NB201, self).get_1gram_count_vector(MY,other.ops,self.OPS_TW)
        
        Mcost = np.asarray([[0,0.2,MAXVAL],[0.2,0,MAXVAL],[MAXVAL,MAXVAL,0]])
        # from Table 1 in https://arxiv.org/pdf/1802.07191.pdf
        
        Wd=ot.emd2(opX,opY,Mcost)
        
        return Wd
    
    
    def gw_distance(self, other):
        return super(Cell_NB201, self).gw_distance(other)

    #def get_1gram_count_vector(self,MX,ops):
        #return super(Cell_NB201, self).get_1gram_count_vector(MX,ops)

    def tw_distance(self, other,lamb=0.5):
        MX=self.matrix
        MY=other.matrix
        #Xops=self.ops[1:-1]
        #Yops=other.ops[1:-1]
        
        #MX=MX[1:-1,1:-1] # crop 7x7 to 5x5
        #MY=MY[1:-1,1:-1] # crop 7x7 to 5x5
        
        # remove empty row and empty col
        opX=self.get_1gram_count_vector(MX,self.ops,self.OPS_TW)
        opY=self.get_1gram_count_vector(MY,other.ops,self.OPS_TW)
    
        #opX = [self.ops.count(op) for op in OPS]
        #opY = [other.ops.count(op) for op in OPS]
        layerX=shortest_path(MX,method="D")
        layerX[layerX==np.inf]=-1
        layerX=layerX[0,:]
        #layerXOut=layerX[:,0]

        layerY=shortest_path(MY,method="D")
        layerY[layerY==np.inf]=-1
        layerY=layerY[0,:]
    
        return TW_NASBENCH201(MX, MY, opX, opY, layerX,layerY)
    
    def mapping_operation(self,opsrow,opscol,OPS):
        
        if opsrow==OPS[0]:# cov3x3
            uu=1
        if opsrow==OPS[1]:# cov 1x1
            uu=2
        if opsrow==OPS[2]:# ave pooling
            uu=3
        if opsrow==OPS[3]:# skip connect
            uu=4
            
        if opscol==OPS[0]:# cov3x3
            index=4*uu
            return index
        if opscol==OPS[1]:#cov 1x1
            index=4*uu+1
            return index
        if opscol==OPS[2]:#ave pooling
            index=4*uu+2
            return index
        if opscol==OPS[3]:#skip connect
            index=4*uu+3
            return index

        return -1
        
    def count_operation_2gram(self,MX,ops):
        count=[0]*20
        
        # first three dimension 1gram
        count[:4]=self.get_1gram_count_vector(MX,ops,self.OPS_TW) # remove None
        
        MX=MX[1:-1,1:-1] # crop 7x7 to 5x5
        ops=ops[1:-1] # remove INPUT, OUTPUT
        
        idx=[ii for ii, val in enumerate(ops) if val in self.OPS_TW]
        temp=MX[idx,:]
        MX=temp[:,idx]
        
        ops = [ops[ii] for ii in idx]
        #ops=ops[idx]
        
        # process 9 remaining dimension
        for ii in range(MX.shape[0]): # each row
            for jj in range(MX.shape[1]): # each column
                if MX[ii,jj]>0:
                    index=self.mapping_operation(ops[ii],ops[jj],self.OPS_TW)
                    count[index]+=1
                
        return count

#    def tw_2gram_distance(self,other,lamb=0.5):
#        return super(Cell_NB201, self).tw_2gram_distance(other,lamb)
        
    def tw_2g_distance(self,other):
        MX=self.matrix
        MY=other.matrix
        
        # remove empty row
        #tempX=np.sum(MX,axis=1)
        #idx= np.argwhere(tempX).ravel()
        #opX=[self.ops[ii] for ii in idx]

        #tempY=np.sum(MY,axis=1)
        #idx= np.argwhere(tempY).ravel()
        #opY=[other.ops[ii] for ii in idx]
        
        
        opX = self.count_operation_2gram(MX,self.ops)
        opY = self.count_operation_2gram(MY,other.ops)
        
        layerX=shortest_path(MX,method="D")
        layerX[layerX==np.inf]=-1
        layerX=layerX[0,:]
        #layerXOut=layerX[:,0]

        layerY=shortest_path(MY,method="D")
        layerY[layerY==np.inf]=-1
        layerY=layerY[0,:]

        return TW_2G_NB201(MX,MY,opX,opY,layerX,layerY) # return 3 elements
                