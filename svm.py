import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import re



#find best C
def svc_param_selection(X, y, nfolds,parameters,svm_model):
    
    
    grid_svm = GridSearchCV(svm_model,parameters, cv=nfolds, scoring="accuracy")
    #fit the search object to input training data
    grid_svm.fit(X, y)
    #return the best parameters
    grid_svm.best_params_
    return grid_svm.best_params_


# Main function that runs the program
def main():
    #A
    raw_data = pd.read_csv("breast-cancer-wisconsin.data", 
                           names = ["id",  "Clump_Thickness", "Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape", 
                                      "Marginal_Adhesion", "Single_Epithelial_Cell_Size","Bare_Nuclei", "Bland_Chromatin",
                                      "Normal_Nucleoli", "Mitoses", "Class"])
    #print(raw_data.shape)
    #print(raw_data.head())

    raw_data["Bare_Nuclei"].replace({"?": None}, inplace=True)
    raw_data['Bare_Nuclei'] = raw_data['Bare_Nuclei'].fillna(raw_data['Bare_Nuclei'].median())
    raw_data['Bare_Nuclei'] = pd.to_numeric(raw_data['Bare_Nuclei'])
    
    # drop ID and Class columns
    raw_data2 = raw_data.drop(['id','Class'], axis=1)
   
    # normalize the data to have a mean of 0 and std deviation of 1 (standard normal distribution)
    # normalize by subtracting raw scores from mean and dividing by std deviation (z-score)
    norm_data = (raw_data2 - np.mean(raw_data2)) / np.std(raw_data2)
    norm_data.head()

    # map class variable to 1's (malignant) and 0's (benign)
    norm_data['Class'] = raw_data['Class'].map({4:1, 2:0})
    norm_data.head()


    #divide normalized data into features and labels
    features = norm_data.drop('Class', axis=1)
    labels = norm_data['Class']
    #print(labels.head())
    features.head()


    # split data into training and test features and labels using 50% of data as validation/test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.5)
    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)
    
    #B
    svm_model = SVC()
    parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},]
    best_c_train=svc_param_selection(X_train, y_train, 20,parameters,svm_model)
    best_c_test=svc_param_selection(X_test, y_test, 20,parameters,svm_model)
    print(best_c_train)
    print(best_c_test)

    #find best C with diagram for linear kernel
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []
    for c in C:
        svc = SVC(C=c, kernel='linear')
        svc.fit(X_train, y_train)
    
        train_acc.append((svc.score(X_train, y_train)))
        test_acc.append((svc.score(X_test, y_test)))

    plt.figure()
    plt.title('SVM Linear Kernel')
    plt.xlabel('C values')
    plt.ylabel('Train Accuracy')
    plt.plot(C, train_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()


    plt.figure()
    plt.title('SVM Linear Kernel')
    plt.xlabel('C values')
    plt.ylabel('Test Accuracy')
    plt.plot(C, test_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()

    #C
    # the same process with Gaussia kernel
    parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.1], 'kernel': ['rbf']},]
    best_c_train=svc_param_selection(X_train, y_train, 20,parameters,svm_model)
    best_c_test=svc_param_selection(X_test, y_test, 20,parameters,svm_model)
    print(best_c_train)
    print(best_c_test)

    #find best C with diagram for Gaussia kernel
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []
    for c in C:
        svc = SVC(C=c, kernel='rbf')
        svc.fit(X_train, y_train)
    
        train_acc.append((svc.score(X_train, y_train)))
        test_acc.append((svc.score(X_test, y_test)))

    plt.figure()
    plt.title('SVM Gaussian Kernel')
    plt.xlabel('C values')
    plt.ylabel('Train Accuracy')
    plt.plot(C, train_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()


    plt.figure()
    plt.title('SVM Gaussian Kernel')
    plt.xlabel('C values')
    plt.ylabel('Test Accuracy')
    plt.plot(C, test_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()

    #D
    parameters = [{'C': [0.1],'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},]
    best_gamma_train=svc_param_selection(X_train, y_train, 20,parameters,svm_model)
    best_gamma_test=svc_param_selection(X_test, y_test, 20,parameters,svm_model)
    print(best_gamma_train)
    print(best_gamma_test)

     #find best gamma with diagram for Gaussia kernel
    gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []
    for g in gamma:
        svc = SVC(C=0.1,gamma=g,kernel='rbf')
        svc.fit(X_train, y_train)
    
        train_acc.append((svc.score(X_train, y_train)))
        test_acc.append((svc.score(X_test, y_test)))

    plt.figure()
    plt.title('SVM Gaussian Kernel')
    plt.xlabel('Gamma values')
    plt.ylabel('Train Accuracy')
    plt.plot(C, train_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()


    plt.figure()
    plt.title('SVM Gaussian Kernel')
    plt.xlabel('Gamma values')
    plt.ylabel('Test Accuracy')
    plt.plot(C, test_acc)
    plt.xscale('log')
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    plt.show()


    

if __name__ == '__main__':
    main()
    
