# NLP-based-Twitter-Sentiment-Analysis

This project explores sentiment analysis on a Twitter dataset using various machine learning models.

## Project Goal

The primary objective is to build and evaluate models capable of accurately classifying the sentiment expressed in tweets. This project serves as an introductory exploration of Natural Language Processing (NLP) techniques.

## Dataset

The dataset used for this project is 'twitter_training.csv' which is a collection of tweets labeled with their corresponding sentiment (positive, negative, neutral).  The dataset is preprocessed, and a TF-IDF vectorizer is used to convert textual data into numerical representations suitable for machine learning models.

## Methodology

1. **Data Exploration:** The project starts with exploratory data analysis to understand the distribution of tweet lengths, sentiment categories, and relationships between different features. Visualizations such as histograms, boxplots, and word clouds are used to gain insights.

2. **Data Preprocessing:** Text preprocessing is a crucial step in NLP. This includes:
    - Converting to lowercase.
    - Removing URLs, non-ASCII characters, and punctuation.
    - Removing stop words.
    - Lemmatization to reduce words to their base forms.

3. **Feature Extraction:**  The TF-IDF vectorizer converts the preprocessed tweets into numerical vectors that capture word frequencies and importance.

4. **Model Training:** Several classification models are trained and evaluated, including:
    - Random Forest
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Artificial Neural Network (ANN)

5. **Model Evaluation:** Performance metrics such as accuracy, precision, recall, F1-score, and confusion matrices are used to compare the models.

6. **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in the dataset.

7. **Result Visualization:** Model accuracy is visually presented through bar charts.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn
- NLTK
- Wordcloud
- Imbalanced-learn

## Project Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt` (Create a requirements.txt file listing all your library dependencies)
3. Mount your Google Drive (if using Google Colab) to access the dataset. 
4. Modify file paths in the code to point to the correct dataset location.


## Future Work

- Further exploration of different NLP techniques.
- Experimentation with deep learning architectures like RNNs or Transformers.
- Hyperparameter tuning to improve model performance.
- Real-world deployment of the model.
