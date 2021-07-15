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

<a href=https://en.wikipedia.org/wiki/Decision_tree_learning>Decision tree algorithm</a>


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
DISCLAIMER:
Unfourtantely we did not manage to find good regression model that will be good enough for predicting the glucose level. even though, we bring here a short summary of the results of the tests that we tried.

Knn Regressor:

We choose the best hyperparameter k which will give the best result.

We measure the error rate of the algorithm with each K using Root Mean Square Error (RMSE).
k=71 will give us the best result in this case.

In the random forest and the decision tree algorithms we choose max_depth = 4. (as we explained above)

running all the algorithms on the same training set and test set:
 |algorithm        |explained_variance_score|max_error         |mean_absolute_error|mean_squared_error|mean_squared_log_error|mean_absolute_percentage_error|median_absolute_error|r2                 |
|-----------------|------------------------|------------------|-------------------|------------------|----------------------|------------------------------|---------------------|-------------------|
|Knn              |0.08644817188435305     |174.96846153846153|32.2110813771518   |1874.7397262643556|0.12227214799280926   |0.3035689361271416            |24.1413076923077     |0.08616970234906707|
|linear regression|0.08264879160517558     |166.1676191238422 |32.70888211352479  |1882.411888555488 |0.12472806843967049   |0.3128261155558476            |25.16249122909972    |0.08242995423795063|
|decision tree    |0.08426725119997935     |169.20203703703703|32.64331939670012  |1879.1715504712588|0.1244036191842194    |0.31207819226471717           |25.13110265313383    |0.08400943701868913|
|random forest    |0.10108666805874822     |160.3477771373162 |32.48402530968287  |1844.4411169822088|0.12216998836340097   |0.3102875885993339            |25.09986139892957    |0.10093857226248404|




 
