import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

def Ncomponents(sorted_eigenvalues, coverage): #returns the number of PCA components given the sorted eigenvalues and the desired coverage
    i = 1
    denominator = sum(sorted_eigenvalues)
    while i >= 1:
        numerator = 0
        for j in range(i):
            numerator += sorted_eigenvalues[j]
            #print('numerator',numerator)
        #print('numerator/denominator', numerator/denominator)
        #print(coverage)
        if (numerator/denominator) >= coverage:
            break
        i += 1
    return i

def PCA(X, lables, coverage):
    training_mean = np.mean(X , axis = 0)
    X_meaned = X - training_mean # mean Centering the data  
    cov_mat = np.cov(X_meaned , rowvar = False) # calculating the covariance matrix of the mean-centered data.
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat) #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    sorted_index = np.argsort(eigen_values)[::-1] #sort the eigenvalues and vectors in descending order
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    #select the eigenvectors that cover 78% of information
    n_components = Ncomponents(sorted_eigenvalues, coverage)
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    eigenvalues_subset = sorted_eigenvalues[0:n_components]
    #Transform the data 
    X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()
    # teta calculation (learning)
    X_reduced = np.insert(X_reduced, 0, lables, axis=1) #add labels on first columns
    #mean of each cluster in the reduced space
    mean_vector = []
    for k in range(6):
        mean_vector.append(np.mean(X_reduced[:,1:][X_reduced[:,0] == k+1]  , axis = 0))
    #teta formula
    teta_vector = []
    for j in range(6):
        for i in range(6):
            teta_vector.append(np.linalg.norm(mean_vector[j] - mean_vector[i]))
    teta = max(teta_vector)
    return X_reduced, eigenvalues_subset, eigenvector_subset, n_components, teta, mean_vector, training_mean

def recognition(Y, lables, eigenvalues_subset, eigenvector_subset, teta, mean_vector, training_mean):
    Y_meaned = Y - training_mean
    Y_reduced = np.dot(eigenvector_subset.transpose(),Y_meaned.transpose()).transpose()
    Y_meaned_rebuilt = np.dot(eigenvector_subset, Y_reduced.transpose()).transpose()
    #calculate epsilon
    epsilon = []
    for i in range(len(Y_reduced)):
        i_distances = []
        for j in range(len(mean_vector)):
            i_distances.append(np.linalg.norm(Y_reduced[i] - mean_vector[j]))
        epsilon.append(i_distances)
    #recalculate data in original space
    Y_meaned_rebuilt = np.dot(eigenvector_subset, Y_reduced.transpose()).transpose()
    #calculate zita
    zita =[]
    for k in range(len(Y_meaned)):
        zita.append(np.linalg.norm(Y_meaned_rebuilt[k] - Y_meaned[k]))
    #applies conditions to perform classification
    result= []
    predict =[]
    right = 0
    wrong = 1 
    for h in range(len(Y_meaned)):
        line =[]
        if zita[h] >= teta:
            line.append("not a position")
        elif zita[h] < teta and teta <= min(epsilon[h]):
            line.append("new position")
        elif zita[h] < teta and min(epsilon[h]) < teta:
            line.append(epsilon[h].index(min(epsilon[h]))+ 1)
            predict.append(epsilon[h].index(min(epsilon[h]))+ 1) 
        line.append(lables[h])
        if line[0] == lables[h]:
            line.append(True)
            right += 1
        else:
            line.append(False)
            wrong += 1
        result.append(line)
    test_result = [(right/len(Y))*100, (wrong/len(Y))*100 ]

    return result, test_result, predict
#plot 1d
def plot1d(X_reduced, llables) :
    X_reduced = np.insert(X_reduced, 0, llables, axis=1)
    df = pd.DataFrame(X_reduced, columns=["Cluster",'Feature1', 'Feature2','Feature3'])
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    e = ax.scatter(x,y, marker="s", c=df["Cluster"], s=20, cmap="RdBu")
    classes = ["1 WALKING", "2 WALKING_UPSTAIRS", "3 WALKING_DOWNSTAIRS", "4 SITTING", "5 STANDING", "6 LAYING"]
    plt.legend(handles=e.legend_elements()[0], labels = classes)
    plt.show()
    return

def plot2d(X_reduced, llables) :
    df = pd.DataFrame(X_reduced, columns=["Cluster",'Feature1', 'Feature2' ])
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    e = ax.scatter(x,y, marker="s", c=df["Cluster"], s=20, cmap="RdBu")
    classes = ["1 WALKING", "2 WALKING_UPSTAIRS", "3 WALKING_DOWNSTAIRS", "4 SITTING", "5 STANDING", "6 LAYING"]
    plt.legend(handles=e.legend_elements()[0], labels = classes)
    plt.show()
    return

def plot3d(X_reduced, llables) :
    df = pd.DataFrame(X_reduced, columns=["Cluster",'Feature1', 'Feature2','Feature3'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    z = np.array(df['Feature3'])
    e = ax.scatter(x,y,z, marker="s", c=df["Cluster"], s=20, cmap="RdBu")
    classes = ["1 WALKING", "2 WALKING_UPSTAIRS", "3 WALKING_DOWNSTAIRS", "4 SITTING", "5 STANDING", "6 LAYING"]
    plt.legend(handles=e.legend_elements()[0], labels = classes)
    plt.show()
    return
    
    

#data extraction from .txt files. IMP change path on an other computer 
data_set = np.loadtxt('C:\\Users\\Administrator\\Documents\\Nicolò\\fondamenti di ML\\UCI HAR Dataset\\train\\X_train.txt')
lables = np.loadtxt('C:\\Users\\Administrator\\Documents\\Nicolò\\fondamenti di ML\\UCI HAR Dataset\\train\\y_train.txt', dtype=int)
test_set = np.loadtxt('C:\\Users\\Administrator\\Documents\\Nicolò\\fondamenti di ML\\UCI HAR Dataset\\test\\X_test.txt')
lables_test = np.loadtxt('C:\\Users\\Administrator\\Documents\\Nicolò\\fondamenti di ML\\UCI HAR Dataset\\test\\y_test.txt', dtype=int)

X_reduced, eigenvalues_subset, eigenvector_subset, n_components, teta, mean_vector, training_mean = PCA(data_set, lables, 0.90)
res, stat, predict = recognition(test_set, lables_test, eigenvalues_subset, eigenvector_subset, teta, mean_vector, training_mean)
print('n dimensions: ', n_components)
print('Right guess: ',stat[0], '%')
print('Wrong guess: ',stat[1], '%')
#plot1d(X_reduced, lables)
#plot2d(X_reduced, lables)
#plot3d(X_reduced, lables)
print('Confusion matrix:')
print(confusion_matrix(lables_test.tolist(), predict))



    

        




