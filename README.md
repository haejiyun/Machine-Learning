# Machine Learning Portfolio

This is collection of my machine learning projects

----------
## [Python] Prediction of Socio-Economic Classes

**Algorithms**: Logistic Regression, Random Forest, SVM<br/>
**Techniques**: Cross-Validation, ROC, AUC<br/>
**Metrics**: Accuracy, F1 Score, Contingency Table, Confusion Matrix<br/>
**Challenges**: Unknown Classes, Imbalanced Data<br/>

Socio-economics is a multifaceted concept that encompasses factors such as income, profession, education, and more, within a society. Although there are no universal socio-economic classifications, distinguishing between socio-economic classes is necessary for various purposes.

In this project, we aim to classify 99,989 individuals in France into two distinct classes, which remain undisclosed for confidentiality purposes. The classification process relies solely on data, ensuring that it is free from any preconceived biases or assumptions regarding the classes. A significant challenge is the imbalanced nature of the classes. The input data includes socio-economic variables such as education level, household type, age, gender, and city of residence.

Classical classification algorithms are fine-tuned to achieve a target precision of 90%.<br/>

<p align="center">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Supervised%20Learning/distribution.png" alt="distribution" width="250" height="200">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Supervised%20Learning/heatmap.png" alt="heatmap" width="300" height="200">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Supervised%20Learning/ROC.png" alt="ROC" width="250" height="200">
<p/>

<a href="https://github.com/haejiyun/Machine-Learning/blob/main/Supervised%20Learning/supervised_learning.pdf">Project</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Supervised%20Learning/supervised_learning.py">Code</a><br/>
<br/>


----------
## [Python] Clustering of French Cities and Rupture Detection of Voting Behavior

**Algorithms**: PCA, t-SNE, UMAP, K-means, PELT<br/>

“Une histoire du conflit” by Thomas Piketty and Julia Cagé explores the evolution of electoral social structures in France. The book serves as a valuable source of socio-economic data over time.

In our project, we utilize data such as voting preferences, average income, average age, educational attainment, home ownership rates, and population to analyze the characteristics of different clusters of areas in France in relation to voting behavior. Furthermore, we identify points where significant changes in voting behavior occur.

As this study employs unsupervised learning techniques, much of the analysis is graphical and visual, allowing us to uncover meaningful insights from the data.

We employ various techniques for dimensionality reduction, clustering, and rupture detection.

<p align="center">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Unsupervised%20Learning/correlation-circle.png" width="350" height="200">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Unsupervised%20Learning/pelt.png" width="400" height="200">
<p/>

<a href="https://github.com/haejiyun/Machine-Learning/blob/main/Unsupervised%20Learning/Unsupervised_Learning_Haeji_YUN.pdf">Project</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Unsupervised%20Learning/Unsupervised_Learning_Haeji_YUN.py">Code</a><br/>
<br/>


----------
## [Python/Deep Learning] Image classification of hollywood actresses

**Algorithms**: VGG, LeNet, EfficientNet

This study aims to classify images of Hollywood actresses to correctly identify each individual. The classification involves three classes: Keira, Nathalie, and Others. To accurately assign the correct class to each image, we fine-tuned several deep learning algorithms. 

Our dataset consists of 429 images for training, and 168 images each for validation and testing. Given the small size of the dataset, data augmentation techniques were applied to enhance the model's performance by artificially increasing the diversity of the training data.

By leveraging these techniques, we aimed to improve the model's generalization and accuracy in recognizing and differentiating between the actresses.

<p align="center">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Deep%20Learning/keira.png" width="120" height="150">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Deep%20Learning/nathalie.png" width="120" height="150">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Deep%20Learning/others.png" width="120" height="150">
<p/>

<a href="https://github.com/haejiyun/Machine-Learning/blob/main/Deep%20Learning/deep-learning-project.ipynb">Code</a><br/>
<br/>


----------
## [Python/Deep Learning] Rakuten France Multimodal Product Data Classification

**Algorithms**: VGG16, Logistic Regression, SVC, Random Forest Classifier, Voting Classifier

Rakuten has issued a challenge on the ENS Challenge data site. The goal is to correctly classify 84,916 products into one of 27 categories on their e-commerce platform, with the weighted F1-score as the performance metric.

We performed extensive data cleaning, processing, and exploration on both text and image data, including natural language processing (NLP) techniques. Both classical machine learning algorithms and deep learning models were applied and fine-tuned to identify the best-performing model.

<a href="https://rakuten-bimodal-classification.streamlit.app/">Project</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Rakuten%20Product%20Classification/1.%20EDA.ipynb">Code-EDA</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Rakuten%20Product%20Classification/2.%20ML.ipynb">Code-Classical Machine Learning</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Rakuten%20Product%20Classification/3.%20Deep.ipynb">Code-Deep Learning</a><br/>
<br/>


----------
## [Python] Country Selection for Chicken Export by Clustering

**Algorithms**: PCA, K-means, Hierarchical Clustering<br/>

When expanding into new markets, selecting the right target countries is crucial for successful and strategic market expansion. This project focuses on identifying optimal countries for chicken exports on a global scale. To achieve this, we conducted an analysis using data related to country fact sheets and food availability provided by the FAO.

Extensive data cleaning and preprocessing were performed, followed by preliminary data exploration. These steps were essential to apply various clustering and dimensionality reduction techniques effectively.

<p align="center">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Sales%20Country%20Clustering/clustering.png" width="700" height="300">
<p/>

<a href="https://github.com/haejiyun/Machine-Learning/blob/main/Sales%20Country%20Clustering/market%20study.pdf">Project</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Sales%20Country%20Clustering/preparation_nettoyage.ipynb">Code-Data Cleaning</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Sales%20Country%20Clustering/clustering_visualisation.ipynb">Code-Clustering</a><br/>
<br/>

----------
## [Python] Fake Bill Detection

**Algorithms**: K-means, Logistic Regression

With the rise in counterfeit currency circulation worldwide, various methods such as analyzing color, texture, and watermarks are used to detect fake bills.

In this project, we focus on detecting counterfeit bills solely based on their geometrical features. Our dataset consists of 1,500 bills, of which 500 are counterfeit. We explore both unsupervised learning through clustering and supervised learning through classification to identify the most effective approach.

<p align="center">
  <img src="https://github.com/haejiyun/Machine-Learning/blob/main/Fake%20Bill%20Detection/variability.png"  width="700" height="200">
<p/>

<a href="https://github.com/haejiyun/Machine-Learning/blob/main/Fake%20Bill%20Detection/detection_faux_billet.pdf">Project</a>; <a href="https://github.com/haejiyun/Machine-Learning/blob/main/Fake%20Bill%20Detection/detection_faux_billet.ipynb">Code</a><br/>
<br/>

