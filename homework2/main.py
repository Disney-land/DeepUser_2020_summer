####################################################################################
# File Name : homework2                                                            #  
# Date  : 2020/07/25                                                               #  
# OS : Windows 10                                                                  #  
# Author : ChanghyunLim                                                            #
# -------------------------------------------------------------------------------  #  
# requirements : python 3.x / numpy / pandas / matplotlib / scikit-learn           #
#                                                                                  #
####################################################################################   
import random                       # import random for random samping first centroids
import numpy as np                  # import numpy for 2d matrix processing
import pandas as pd                 # pandas for overall processing
import matplotlib.pyplot as plt     # for showing plots
import warnings                     # defrecation issue handler

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.")  # print general error  
    pass

LIMITION = 900  # Limit the maximum iteration(for get result much faster)
seed_num = 777  # set random seed
np.random.seed(seed_num) # seed setting
iteration = 300 # if value unchage untill 300 times

class kmeans_:
    def __init__(self, k, data, iteration): # initalize
        self.k = k # number of cluster
        self.data = data    # data
        self.iteration = iteration # set iteration [300]

    def Centroids(self, ):  # Set initial centorids
        data = self.data.to_numpy() # get data and change numpy for sampling
        idx = np.random.randint(np.size(data,0), size=self.k) # get random index
        sampled_cen = data[idx,:] # sampling
        return sampled_cen # return init centers

    def Assignment(self, ): # code for overall process
        data = self.data.to_numpy() # change data to numpy for processing
        cen = self.Centroids() # get initial centroids
        prev_centro = [[] for i in range(self.k)] # get prev_centroids for compraring
        iters = 0 # set flag 0
        warnings.simplefilter(action="ignore", category=FutureWarning) # ignore warnings check return part of Update
        while self.Update(cen, prev_centro, iters) is not True: #check Update (for LIMITION)
            iters = iters + 1
            clusters = [[] for i in range(self.k)] # set cluster
            old_result = [[] for i in range(self.k)] # set prev_cluster
            clusters = self.get_UD(data, cen, clusters) # set cluster (part of Assignment function)
            idx = 0 # set index
            for result in clusters: # for whole clusters
                prev_centro[idx] = cen[idx] # update centroids
                cen[idx] = np.mean(result, axis=0).tolist() # Get center mean
                idx = idx + 1 # update index counter
            if np.array_equal(old_result, result) is True: # Comparing
                iters = 0 # start from ground again
            iteration = self.iteration # get iteration
            old_result = result # update result                
        return clusters , iteration #return clusters and iterations

    def Update(self,centroids, prev_centro, iters): # Update as a teration checker and centroid assignment
        if iters > LIMITION: # compare for LIMITION
            return True
        warnings.simplefilter(action="ignore", category=FutureWarning) # ignore warnings check return part of Update
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        return prev_centro == centroids
    
    def Train(self, ):  # Train for get result and Processing overall kmeans workings
        iteration = 0 # set iteration 0
        result, iteration = self.Assignment() # get result and iteration
        self.iteration = iteration # update iteration
        return result # return result

    def get_UD(self,data, centroids, clusters): # Get Uclidient distance
        for ins in data:
            mu = min([(i[0], np.linalg.norm(ins-centroids[i[0]])) \
                        for i in enumerate(centroids)], key = lambda t:t[1])[0] # Get uclidient distance
            try: # exception processing
                clusters[mu].append(ins) # for all clusters append instance (as a assignment function)
            except KeyError: # exception handling
                cluster[mu] = [ins] # update case
        for result in clusters: # for all sub-clusters
            if not result: # Nan case
                result.append(data[np.random.randint(0, len(data), size = 1)].flatten().tolist()) # Samplikng and append sub-clusters
        return clusters     # return whole clustsers k sub-clusters

if __name__ == '__main__': # Start from main
    colorlist = ['r','c','k','g','m','b','y'] # Set color list (set this pallet because white and yellow is hard to congize)
    data = pd.read_csv('./data.csv') # load data
    model1 = kmeans_(k=3, data=data, iteration=iteration) # implemented model init setting
    for i in range(model1.k):
        result = np.array(model1.Train()[i-1])
        result_x = result[:,0]
        result_y = result[:,1]
        plt.scatter(result_x,result_y,c=str((colorlist[i]))) #plt scatter for each clusters
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("implementaion") # set title
    plt.show() # show plot

    model2 = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data) # sklearn model init setting
    predict = pd.DataFrame(model2.predict(data)) # update predict label
    predict.columns = ["predict"] # Set col name
    data = pd.concat([data,predict],axis=1) # concat data
    predict.columns=['predict'] # Set col name
    plt.scatter(data['Sepal width'],data['Sepal length'],c=data['predict'],alpha=0.5) # scatter plot
    plt.xlabel('sepal length (cm)') # set label
    plt.ylabel('sepal width (cm)') # set label
    plt.title("from scikit-learn library") # set title
    plt.show() # show plot