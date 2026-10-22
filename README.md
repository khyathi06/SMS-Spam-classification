# SMS-Spam-classification
Text classification using TF-IDF and Decision Tree, KNN, and SVM algorithms
1. Abstract
Text classification is an important task in Natural Language Processing (NLP) that involves assigning predefined categories to textual data. SMS messages are commonly used for communication, but unwanted promotional messages and fraudulent texts create problems for users. Automated spam detection helps identify such messages efficiently.
This project implements an SMS Spam Classification system using three supervised machine learning algorithms: Decision Tree, K-Nearest Neighbours (KNN), and Support Vector Machine (SVM). The SMS Spam Collection dataset is used to classify messages into two categories: Spam and Ham. Text preprocessing and Term Frequency–Inverse Document Frequency (TF-IDF) are used to transform messages into numerical features suitable for machine learning.
The models are evaluated using Accuracy, Precision, Recall, and F1-score. Confusion matrices and comparison graphs are used to analyse the classification results. A what-if experiment is performed by varying the K value in KNN. The most suitable classifier is recommended based on the actual experimental results.

2. Introduction
With the rapid growth of mobile communication, SMS has become an important medium for personal communication, banking notifications, promotional offers, and service alerts. However, users also receive unwanted advertisements, suspicious links, and potentially fraudulent messages.
Manually identifying spam messages is time-consuming. Machine learning provides an automated approach by learning patterns from previously labelled messages and using those patterns to classify new messages.
Text classification converts unstructured text into a form that machine learning algorithms can process. In this project, TF-IDF is used for feature extraction, and three classification algorithms are compared using the same dataset and evaluation procedure.

3. Problem Statement
The objective of this project is to develop an automated system that classifies SMS messages as Spam or Ham. The system uses TF-IDF features and compares Decision Tree, K-Nearest Neighbours, and Support Vector Machine algorithms.
The project aims to determine how effectively each algorithm identifies spam messages while avoiding the incorrect classification of legitimate messages.

4. Objectives
- To understand the concepts of text classification and supervised machine learning.
- To collect and load a labelled SMS dataset.
- To perform data cleaning and text preprocessing.
- To transform text messages into numerical features using TF-IDF.
- To implement Decision Tree, KNN, and SVM classifiers.
- To divide the dataset into training, validation, and testing subsets.
- To evaluate models using Accuracy, Precision, Recall, and F1-score.
- To visualize and compare model performance.
- To analyse confusion matrices and classification errors.
- To conduct a what-if experiment by changing the K value in KNN.
- To recommend a suitable classifier based on actual experimental results.

5. Dataset Description
The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.
Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
The dataset contains SMS messages labelled according to their category.
Ham: A legitimate message that is not classified as spam.
Spam: An unwanted message, such as an unsolicited advertisement or suspicious promotional message.
The dataset includes two main fields:
- "label": The class of the message.
- "message": The text content of the SMS.
The dataset initially contains 5,572 messages. Missing values and duplicate records are handled during preprocessing. Therefore, the final number of samples used for modelling may be lower than the initial count.

6. Theoretical Background
6.1 Natural Language Processing
Natural Language Processing is a branch of Artificial Intelligence that enables computers to process, analyse, and understand human language. Text classification is one of its applications.
Other NLP applications include sentiment analysis, spam detection, language translation, document categorization, and chatbot development.
In this project, NLP techniques are used to represent SMS messages numerically so that machine learning algorithms can classify them.
6.2 Supervised Machine Learning
Supervised learning is a machine learning approach in which a model learns from labelled examples. Each training example contains input data and its corresponding target label.
For SMS spam classification, the input is the message text and the target is either Spam or Ham. The model learns relationships between the text features and labels and uses them to predict the class of unseen messages.
6.3 Text Preprocessing
Text preprocessing prepares raw messages for feature extraction and classification. It helps maintain consistent input and removes data quality problems.
The preprocessing in this project includes:
- Checking for missing values.
- Removing duplicate records.
- Converting class labels into numerical values.
- Converting text to lowercase through the TF-IDF vectorizer.
- Excluding common English stop words according to the vectorizer configuration.
The target labels are represented as Ham = 0 and Spam = 1.
6.4 TF-IDF Feature Extraction
TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a feature extraction technique that represents text documents as numerical vectors.
Term Frequency measures how often a word occurs in a document. Inverse Document Frequency reduces the importance of words that appear in many documents.
A simplified representation is:
TF-IDF(t,d) = TF(t,d) × IDF(t)
where "t" represents a term and "d" represents a document.
Words that are more informative for distinguishing documents may receive higher TF-IDF weights. These numerical features allow machine learning algorithms to process text data.
In this project, the vectorizer is fitted only on the training data and is then applied to the validation and test data. The vocabulary is limited to a maximum of 3,000 features.

7. Machine Learning Algorithms
7.1 Decision Tree Classifier
A Decision Tree is a supervised machine learning algorithm that classifies data using a sequence of decision rules. The algorithm divides data into smaller groups based on feature values until a prediction can be made at a leaf node.
For text classification, the features are TF-IDF values. The model learns splits that help distinguish spam messages from legitimate messages.
Advantages:
- Easy to understand and interpret.
- Can model nonlinear decision boundaries.
- Requires little feature scaling.
Limitations:
- May overfit the training data.
- Can be sensitive to changes in the training dataset.
- May struggle with high-dimensional sparse text features.
7.2 K-Nearest Neighbours (KNN)
K-Nearest Neighbours is a supervised learning algorithm that predicts the class of a new sample based on the labels of its nearest training samples.
The parameter K determines the number of neighbours considered. For example, when K = 5, the algorithm considers five nearby training samples and predicts the class based on their majority vote.
Advantages:
- Simple to understand and implement.
- Does not require a complex model-fitting process.
- Performance can be explored by varying K.
Limitations:
- Prediction can be computationally expensive.
- Results depend on the distance measure and the selected K value.
- High-dimensional text features can make distance-based comparisons less effective.
7.3 Support Vector Machine (SVM)
A Support Vector Machine is a supervised learning algorithm that finds a decision boundary separating different classes. A linear SVM aims to find a hyperplane with a large margin between the classes.
SVM is often effective for text classification because TF-IDF data usually contains many features, most of which are zero.
Advantages:
- Often effective for high-dimensional text data.
- Can generalize well when appropriately configured.
- Linear prediction is efficient for sparse text vectors.
Limitations:
- The model is less interpretable than a simple decision tree.
- Results depend on parameter settings.
- Class imbalance and overlapping examples can affect predictions.

8. Dataset Splitting and Experimental Design
The dataset is divided into three subsets:
- Training set — 80%: Used to train the classifiers.
- Validation set — 10%: Used to compare models and examine parameter choices.
- Testing set — 10%: Used for final performance evaluation.
Stratified splitting is used to preserve the approximate proportions of Spam and Ham messages in each subset.
All three algorithms use the same data partitions and TF-IDF representation to make the comparison fair. The test dataset should remain untouched during model selection and parameter tuning.

9. Evaluation Metrics
9.1 Accuracy
Accuracy measures the proportion of all predictions that are correct.
Accuracy = (TP + TN) / (TP + TN + FP + FN)
9.2 Precision
Precision measures the proportion of messages predicted as spam that are actually spam.
Precision = TP / (TP + FP)
High precision means fewer legitimate messages are incorrectly flagged as spam.
9.3 Recall
Recall measures the proportion of actual spam messages that are correctly identified.
Recall = TP / (TP + FN)
High recall means fewer spam messages are missed.
9.4 F1-score
F1-score is the harmonic mean of Precision and Recall.
F1-score = 2 × (Precision × Recall) / (Precision + Recall)
The F1-score is useful when both false alarms and missed spam messages matter, especially when the classes are imbalanced.
In these formulas, Spam is treated as the positive class, TP means True Positive, TN means True Negative, FP means False Positive, and FN means False Negative.

10. Confusion Matrix
A confusion matrix compares actual labels with predicted labels.
- True Positive (TP): A spam message correctly classified as spam.
- True Negative (TN): A ham message correctly classified as ham.
- False Positive (FP): A ham message incorrectly classified as spam.
- False Negative (FN): A spam message incorrectly classified as ham.
The confusion matrix helps identify the types of errors made by each model. In spam detection, false positives can cause important legitimate messages to be missed, while false negatives allow unwanted spam to reach the user.

11. Comparative Analysis
Decision Tree, KNN, and SVM are evaluated using Accuracy, Precision, Recall, and F1-score. Their results are presented in a comparative table and a graph generated from the experimental outputs.
The analysis considers:
- Which algorithm achieves the highest accuracy.
- Which model identifies spam more effectively.
- Whether a model incorrectly flags legitimate messages.
- Which model provides the best balance between Precision and Recall.
- The training and prediction time of each classifier.
- The trade-off between performance and computational cost.
No classifier should be declared the best before examining the actual experimental results.

12. What-if Analysis: Effect of K in KNN
A what-if experiment examines how changing a model parameter affects classification performance.
In this project, the K value of KNN is varied across K = 3, 5, 7, and 9. Each configuration is trained on the same training data and evaluated using the same validation set.
A smaller K value may respond more strongly to individual training examples and noise. A larger K value may produce smoother decisions but can overlook local patterns.
Accuracy and F1-score are compared for each value of K. The results help identify how the parameter affects performance on this dataset.
The best K value should be selected using validation results, and the final test set should be used only for the final evaluation.

13. Tools and Technologies
- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF Vectorizer
- GitHub

14. Applications
SMS spam classification can be used in:
- Mobile messaging applications.
- Automated spam filtering systems.
- Fraudulent message screening.
- Telecommunications services.
- Message categorization and content filtering.
The model is an educational prototype and should not be treated as a complete fraud detection or security system without further testing.

15. Limitations
- The dataset may not represent all current SMS spam patterns.
- New spam messages can use unfamiliar words or techniques.
- Text-only features may not capture sender reputation, links, or other contextual information.
- Performance may change when the model is applied to data from a different source.
- False positives and false negatives can still occur.

16. Future Enhancements
- Evaluate additional text representations such as word n-grams or character n-grams.
- Test more classification algorithms.
- Tune model parameters using validation data.
- Evaluate the model on newer and more diverse SMS datasets.
- Develop a simple web application for testing individual messages.
- Explore methods for handling class imbalance.
- Improve detection of emerging spam patterns.

17. Conclusion
This project demonstrates an end-to-end text classification workflow using an SMS dataset, preprocessing, TF-IDF feature extraction, and three supervised machine learning algorithms: Decision Tree, KNN, and SVM.
The models are evaluated using Accuracy, Precision, Recall, F1-score, confusion matrices, and comparative visualizations. A what-if experiment examines the effect of different K values on KNN performance.
The final recommendation is based on the measured experimental results, including classification performance and computational considerations. The project provides practical experience in NLP, machine learning evaluation, data visualization, and reproducible experimentation.

18. References
1. UCI Machine Learning Repository, SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. Pandas Documentation: https://pandas.pydata.org/docs/
4. Matplotlib Documentation: https://matplotlib.org/stable/
5. Seaborn Documentation: https://seaborn.pydata.org/
