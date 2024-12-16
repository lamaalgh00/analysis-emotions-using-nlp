#Arabic Emotion Analysis using Machine Learning and Deep Learning
This project is a group collaboration focused on analyzing emotions in Arabic tweets using both machine learning (ML) and deep learning (DL) approaches.
The dataset used in this project is an Arabic Emotions Twitter Dataset, containing over 10,000 tweets. It is specifically curated to represent the most commonly encountered emotion categories in Arabic tweets. The dataset consists of two columns:
Tweets (text data)
Labels (numeric, ranging from 0 to 7)
Emotion Labels:
Label	Emotion
0:	None
1:	Anger
2:	Joy
3:	Sadness
4:	Love
5:	Sympathy
6:	Surprise
7:	Fear

Data was split into 70% training and 30% testing subsets.

Machine Learning Models
- Model 1: Logistic Regression with CountVectorizer
Text data was transformed into word count-based features using CountVectorizer.
A logistic regression classifier was trained to predict emotion labels based on these features.
- Model 2: Linear Support Vector Classifier (SVC) with TfidfVectorizer
Text data was transformed into TF-IDF features using TfidfVectorizer, capturing the importance of words in the dataset.
A linear SVC classifier was trained to predict emotion labels based on these TF-IDF features.

Deep Learning Model
Sequential Neural Network
- A Sequential model was utilized for deep learning with the following architecture:
- Embedding Layer: Converts input words into dense vector representations.
- Dropout Layers: Reduces overfitting by randomly disabling neurons during training.
- Convolutional Layer: Captures spatial patterns in text data.
- Max Pooling Layer: Reduces feature dimensions while retaining key information.
- Dense Layers: Processes high-level features for classification.
- Output Dense Layer: Predicts the emotion category with softmax activation.

Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Batch Size: 1024
Epochs: 25
