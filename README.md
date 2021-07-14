#  machine learning project

Dataset description: 
The dataset contains medical information about patients, like gender, age, various diseases, and smoking status.  
Each row in the data provides relevant attributes about the patient.

Dataset source: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

Our questions:
1) predict whether a patient is likely to get a stroke based on the input parameters.                                                                     2) predict blood glucose level a patient is likely to have based on the input parameters.
3) predict whether a patient is likely to have a hypertension based on the input parameters
4) predict whether a patient is likely to have a heart disease based on the input parameters

# Data Preprocessing
Data preprocessing is one of the most important things, and it is one of the common factors of success of a model.
We took care of a few things:

* Splitting of the data set in Training and Validation sets
* Taking care of Missing values
* Taking care of Categorical Features
* Treating outliers
* Normalization of data set

# techniques we use:

<a href=https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>k-Nearest-Neighbor</a>

<a href=https://en.wikipedia.org/wiki/Naive_Bayes_classifier>Naive Bayes</a>

<a href=https://en.wikipedia.org/wiki/Random_forest>Random Forest</a>

<a href=https://en.wikipedia.org/wiki/Decision_tree_learning>decision tree algorithm</a>


# results:

In each algorithm we split the data into a 75% training set and 25% test set.  

k-Nearest-Neighbor:

In each question, We choose the favorable hyperparameter k which will give the best result according to a method we found.

source - https://arxiv.org/pdf/1409.0919.pdf 

decision tree algorithm: 

We try to find the favorable criteria for calculating information gain.
Decision tree algorithms use information gain to split a node.   
But there is not much performance difference when using gini index compared to entropy as a splitting criterion.
max_depth: We found that the favorable max_depth in decision tree model is 4.
 
Naive Bayes:

We found that Multinomial Naive Bayes has much better accuracy when compared with the Gaussian technique given the same dataset and parameters.
Multinomial average: 0.93 , Gaussian average: 0.83

Random forest:

we use 100 trees in the algorithm as default. because value greater than 100 slow down the computation.

Comparison of all algorithms:

We've made a comparison of all the algorithms we mentioned above: 
The algorithms are fit and evaluated on the same subsets of the dataset.    
In every question: it chooses the best k using the ‘check_best_k_for_knn’ function for the knn algorithm.   
In the decision tree algorithm,we used the default criterion - gini.
(as we already explained : there is not a big difference between gini and entropy accuracy.)  
And used with max_depth = 4.
In Naive Bayes: we used the multinomial Naive Bayes classifier that suits our data.


<img src="/images/classification_algorithms _results.jpeg" alt="classification_algorithms_results" height="250" width="350" >


As we can see, all of the algorithms have similar accuracy.

# Predicting glucose level

Knn Regressor:

We choose the best hyperparameter k which will give the best result.

We measure the error rate of the algorithm with each K using Root Mean Square Error (RMSE).
k=71 will give us the best result in this case.

In the random forest and the decision tree algorithms we choose max_depth = 4. (as we explained above)

running all the algorithms on the same training set and test set:



 
