# introduction-to-different-Classifiers
# SIX diffenret Classifiers for six sample dataset


## Code Implementation 
---
The program is object-oriented using python 3.7, and we also used the scikit-learn library [1] (0.23.0 version) to implement machine learning methods. The Dataset object contains a list of tuples to store names of the datasets as well as the data as a Pandas data frame (1.1.4 version). By iterating through the Dataset object, we can create another object named “ML-Methods” to perform the desired task on each dataset. In the first step, we used 80% of the data for training and 20% for testing. Moreover, we applied a min-max scaler on datasets to simplify the training procedure. ML-Methods object consists of a list of models as tuples, where each tuple stores one specific machine learning model and its associated name. Thus, by iterating through the list of models, we can apply every method to each dataset while storing the informative data throughout the procedure. We used StratifiedKFold to perform 10-fold cross-validation and report the mean accuracy of all the performances and other desired metrics. Finally, we utilized matplotlib [2](3.3.3 version) to plot the accuracy of each method on each dataset using the test dataset. Furthermore, the decision boundary for each method is also shown in separate plots. Finally, the source code, including all the generated plots are available in a GitHub repository.
Accuracy measure for each classifier


![alt text](./readme%20images/table%201.png)

## Discussions and comparisons

In this Program, we implemented four different machine learning models which are going to be discussed in what follows.
Linear discriminant analysis (LDA)
Linear Discriminant analysis tries to reduce dimension to 1D from 2D in case of binary classification features before classifying them. By doing so, it also makes sure that when data points are projected on 1D line, the mean of individual class remains far as possible and scatter or variance between data points of each sample stays minimize.


![alt text](./readme%20images/figure%201.png)

	When trying to use this technique on circles0.3 it made classification more difficult, and model was producing poor results. This is because there would be lots of samples overlapping each other when we project this on 1D line. Similarly, the points pattern of half kernel is somewhat similar to circles0.3. So, by applying the same logic we can guess that model would not be worthy to use in half kernel dataset. 

	Followed by spiral1 with a good accuracy score of 0.77. Observing the sample plot of twogaussians33 and twogaussians42 dataset we can see that LDA would easily find the best line to separate the clusters from each other with maximizing distance of mean points. So, both models gave the best performance while testing with test sets. 
	
	At last, while using LDA on moons it gave 0.92 testing accuracy. This is because some points having opposite class where overlapping when dimensions were reduced but most of remains separated. 
### Quadratic Discriminant Analysis (QDA)

This method is a variant of LDA, which seeks non-linear separation between various classes. Its initial assumption is that the data points have a Gaussian distribution. Despite the LDA that considered each class having a shared covariance matrix, QDA computes a covariance matrix for each class individually. Thus, when the distribution is Gaussian and the assumption of the shared covariance matrix between classes is inaccurate, QDA classifies the points with considerable accuracy. For instance, as it shows in Table 1, QDA achieves high accuracy on twogaussians33 (99%) and circles0.3 (99%) where overlapping data points are not noticeable. Furthermore, QDA performs acceptably on twoguassian42 (95%) and halfkernel (93%) datasets where some overlapping data points existed. However, covariance matrices will highly intersect when the distribution is not gaussian, and data points have overlapping variances on the x and y axes. Therefore, an appropriate curve cannot be found to separate the two classes ideally. As shown in Figure 2, the decision boundaries of QDA do not correctly separate the classes in spiral1(77%) and moon1 (92%) datasets and lead to inaccurate classification.
      

![alt text](./readme%20images/figure%202.png)

### Naive Bayes
There are multiple distributions that are implemented by the Naive Bayes-based methods, the Gaussian, Bernoulli, and Multinomial. By implementing all three types of naive Bayes, as expected, the naive Bayes with normal distribution works more efficiently. Its accuracy level also hits more than 99% for the Circles0.3 and twoGaussians33 datasets. However, Bernoulli naive could not identify a boundary to distinguish classes in spiral1 and moon1 datasets. Its accuracy level was just 75% among all other datasets. 
Naive Bayes assumes that the features are independent. This means we are still assuming class-specific covariance matrices (as in QDA), but the covariance matrices are diagonal. This is due to the assumption that the features are independent. Also, it assumes that the class-conditional densities are normally distributed. Figure 3 shows the decision boundaries of this classifier on each dataset.  
Bernoulli Naive Bayes (BNB) is another assessment technique used in this assignment. Given that Bernoulli distribution is just to identify whether a particular sample is related to a specific class or not, this classification is indeed just able to identify one class. Thus, the accuracy level and all other performance measurements should be nearly 50%.
The Multinomial Naive based method is widely used in text classification when the number of features is too large and we aim to identify what is the text for, or if an email is spam or not. The performance measurements for all mentioned methods are shown in the tables based on their datasets.¬¬¬
      

Figure 3 Naive Bayes decision boundaries for different datasets.
K-nearest neighbor
KNN works on the “plurality vote” concept. It works on the assumption that similar points are near to each other. By selecting the number of neighbors, we take the nearest neighbors and assign the class of the majority to the new data point. We used numerical and graphical approaches to seek the proper K value.
Numerical approach:
x1	x2	Euclidean Dist.	label
0.547081	0.707805	0.213072147	1
0.738788	0.563298	0.247034833	1
0.597446	0.768834	0.285950414	1
Table 2. Numerical approach for choosing the best K
Table 2 shows few sample points of 'twogaussians42' dataset with corresponded Euclidean distance to demonstrate the numerical approach. In this approach, to predict the label for a test point like (t1=0.5,\ t2=0.5), we need to calculate the Euclidean distances from the given point to each point using \ \sqrt{\left(x1-t1\right)^2+\left(x2-t2\right)^2} formula and order them in ascending manner. Then, we counted the labels for various 'K' values. As a result, label “1” was subsequently assigned to the test data point because all data points nearby up to 227 had the same label.

Graphical approach:
 
Figure 4. Accuracy vs K value

We used the same dataset to illustrate the graphical approach. We applied the KNN classifier on 'twogaussians42' dataset for k=1 to 20 and produced the scatter-plot graph based on the accuracy metrics. At k=10, the classifier achieves the greatest accuracy of 96.5%. Similarly, Table 3 shows Optimal K values for each dataset.

Column1	Circles0.3	half kernel	twogaussians33	twogaussians42	spiral1	moons1
Optimal K Value	     All	       All	8	10	18	 2
Accuracy	100	   100	99.5	96.5	100	100
Table 3. Optimal K value for each dataset
Performance of KNN with different datasets:

      
Figure 5 KNN decision boundaries for different datasets.
	With the circular and half-kernel datasets, KNN achieves a flawless accuracy of 100%. KNN can categorise these shapes perfectly because it is relying on point localization based on distance.
	Similarly, it can classify the moons and spiral datasets quite accurately; but some outlier data points diminish the majority vote in some locations, making it slightly less accurate than what was observed above.
	Due to the close clustering of data points in Gaussian datasets, KNN cannot categorise them as accurately as in other datasets.
Conclusion
As datapoints of the dataset forms geometric shapes, KNN performs the overall best classifier for all the datasets because of its ability to classify 2-D shapes based on localized data points. For shapes such as circles, half-kernel & spiral, LDA cannot find a linear separation due to their overlapping data points in a shared covariance matrix. However, it can classify the gaussian datasets linearly. Moreover, QDA can accurately classify circles & twogaussians33 dataset, but it lacks to separate classes in spiral dataset due to overlapping covariance matrices and non-gaussian distribution. Finally, Like QDA, Naive Bayes performs well on almost all shapes except Spiral. 
It can be concluded that classification of a dataset in 2-D depends on various features such as the type of distribution, the shape formed by the datapoints, the level of overlapping data points between classes and the algorithms’ ability to separate the classes considering various features.
 
Appendix

Performance measures for each dataset corresponding to the models

 
Table 4. performance measures on the Circles0.3 dataset


 
Table 5. Performance measures on the halfkernel dataset.


 
Table 6. Performance measures on the moons1 dataset


 
Table 7. Performance measure on the Spiral1 dataset


 
Table 8. Performance measure on the twoguassion33 dataset.


 
Table 9. performance measure on the twogaussions42 dataset


The output of the program

 
Figure 6. Screenshot from the output of the program
 
Decision boundary for each classifier

 
Figure 7. Decision boundary for each classifier

References

[1]	D. K. Barupal and O. Fiehn, “Scikit-learn: Machine Learning in Python,” J. ofMachine Learn. Res., vol. 12, pp. 2825–2830, 2011, doi: 10.1289/EHP4713.
[2]	J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” Comput. Sci. Eng., vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.

