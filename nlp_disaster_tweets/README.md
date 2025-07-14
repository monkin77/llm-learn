# NLP - Classifying Tweets for Disaster Response
This notebook contains solutions for the Kaggle competition "NLP - Classifying Tweets for Disaster Response". The goal is to classify tweets as disaster-related or not, exploring various NLP techniques and architectures, such as BERT, LSTMs, Transformers, and more.

# Kaggle Competition
You can find the competition details and dataset on Kaggle:
[https://www.kaggle.com/c/nlp-getting-started](https://www.kaggle.com/c/nlp-getting-started).

# Getting Started
1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Ensure you have the required libraries installed. The parent directory contains a `pyproject.toml` file that can be used to install dependencies using `Poetry`.
3. **Fetch the Dataset**: Download the dataset from Kaggle and place it in the `input` directory.
4. **Run the Notebook**: Open the notebook in Jupyter or any compatible environment and execute the cells sequentially.

# Solutions
1. **DistilBert**: A solution using DistilBERT, a smaller and faster version of the Bidirectional Encoder Representations from Transformers (BERT) model. This is an Encoder-only Architecture that is efficient for text classification tasks.
2. **DistilBert With Ensemble Learning**: This solution builds upon the DistilBERT model by incorporating ensemble learning techniques to improve classification accuracy. Uses K-Fold Cross-Validation to train multiple models and average their predictions.