####################################################################################
# File Name : homework1                                                            #  
# Date  : 2020/07/18                                                               #  
# OS : Windows 10                                                                  #  
# Author : ChanghyunLim                                                            #
# -------------------------------------------------------------------------------  #  
# requirements : python 3.x / numpy / pandas / matplotlib / scikit-learn           #
#                                                                                  #
####################################################################################   

import random                      
import numpy as np                 
import pandas as pd                 
import matplotlib.pyplot as plt    
import warnings                     

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.") 
    pass

if __name__ == '__main__': # Start from main
    data = pd.read_csv("./data.csv") # read the selected csv file
    plt.scatter(data['Sepal width'],data['Sepal length'], color = 'b') # scattering plot
    plt.xlabel('Sepal length(cm)')
    plt.ylabel('Sepal width(cm)')
    plt.title('from scikit-learn library')
    plt.show() # show plot