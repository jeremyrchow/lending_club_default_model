# Predicting Lending Club Loan Defaults
**Jeremy Chow**

The goal of this project was to classify predicting defaulting loans based on pre-loan metrics.

## The Data and Preprocessing
The original data set came from [LendingClub](https://www.lendingclub.com/info/download-data.action). Each row of data consisted of 145 rows with loan data (amount, monthly payment, time issued), account information, settlement information, hardship details, investor funding information. There is a class imbalance of about 80% paid off loans and 20% defulated loans. The first step was to drop all data that would not be available at time of issuance to prevent data leakage. 

## Exploratory Data Analysis

Using the Random Forest Classifier, we can look at feature importance for predicting whether a loan will default:

![Feature Importance](https://github.com/jeremyrchow/lending_club_default_model/blob/master/graphs/feature_importance.png?raw=true)

We can see what random forest believes is the most important features, but note that they only have a 5% max importance, so it seems that the weight of the decisionmaking is evenly distributed between many variables and there's no one strong indicating feature that is the backbone of our model.

## Modeling
First we split the data into an 80%-20% train test split, maintaining the 82%~18% class imbalance between the two new samples. 
We then train three classifier models: [Gradient boosting classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), and [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) against classifying just the 'sub grade' feature, which is Lending Club's metric for estimating the risk of a loan.

The Gradient Boosting Classifier (GBC) model has the best ROC-AUC score of 0.7255, although the models perform similarly. 

![ROC curves comparison between all models](https://github.com/jeremyrchow/lending_club_default_model/blob/master/graphs/ROC_curves_all.png?raw=true)

## Results
Using the GBC model, we can generate the confusion matrices by optimizing for F1 Score.

![Confusion Matrix](https://raw.githubusercontent.com/jeremyrchow/lending_club_default_model/master/graphs/confusion_matrix_rel_actual.png)

The confusion matrix tells us that our model classified 73% of the issued loans as defaulted but at the cost of incorrectly classifying 41% of the paid off loans as defaulted. In the context of this problem, we can deduce how much this would cost by estimating the average cost of a defaulted loan vs. the profit of a successful loan.

## Custom Cost metric

By summing the average money gained from a paid off loan and subtracting the average money lost on a misclassified loan, w e can optimize the threshold for profits. 

However, this strategy yields that of the data set in the first place - in other words, the threshold was set to 0 and none of the issued loans were considered defaulted because accepting all loans in the data set maximized profits.
## Conclusion

The classifier models built were able to identify defaulting loans but at the cost of significant additional false positives that ultimately yielded in lower profits. Thus, LendingClub's strategy seems to be effective if profit is the primary goal.  

## Future Work
To improve on this model, we can try the following:
- Downsample the paid off loans to reach a 50-50 split
- Optimize models using more gridsearch of parameters (ie. finding optimal leaves for random forest)
