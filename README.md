# Machine-Learning-examples
A couple of machine learning examples are shown using different classifiers and data-split techniques.

- **Compare classifier hyperplates**: Comparing classifier hyperplates by visualising the Test set results. Classifier examples:
  * SVM with rbf and linear kernel
  * Adaboost
  * KNN
  * Random forest
 - **Multiple_regression**: Simple multiple regression template using linear regression
 - **SVM_Random_split**: SVM random split template
 - **SVM_kfold**: SVM k-fold cross validation template
 - **Sequential_DNN_template_K-fold_final**: Sequential DNN template for k-fold cross validation. The template provides a binary classification, with a confusion matrix. Metrics given: Accuracy; For each class: Recall, Precision, F1 score, Specificity, Miss rate, Fall-out, False discovery rate, False omisson rate, Negative predictive value. ROC curve.
 - **Fully-Connected Deep Neural Network_k-fold_Updated**: My updated template for a binary classification, with a confusion matrix. Metrix given: Accuracy, Matthews Correlation Coefficient. For each class: Recall, Precision, F1 score, Specificity, Sensitivity
  - **Fully-Connected Deep Neural Network_random_split_Updated**: Simirarly, a template for a binary classification using random split.

## Helper function
The file *custom_metrics.py* contains a range of useful functions that one could use, such as getting the accuracy, recall, precision, f1score, specificity, sensitivity, Matthews Correlation Coefficient (MCC), can print the confusion matrix in a pretty way (as wikipedia defines it and not like sklearn does it), and calculates the Pearson coeff, Spearman coeff, MIC & Cosine Similarity

Lazy update: there's a bug in the DNN notebook, will update it later maybe


