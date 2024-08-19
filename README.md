# Criterion Collection Recommendations

## Overview

This project is designed to recommend movies from the Criterion Collection based on user preferences. The recommendation system leverages various data processing techniques, including topic modeling, keyword extraction, sentiment analysis, and similarity calculations. The goal is to provide users with personalized movie suggestions by analyzing and comparing features from a dataset of Criterion Collection movies.

## Features

- **Data Cleaning & Preprocessing**: Handles missing values and performs text preprocessing to prepare the dataset for analysis.
- **Topic Modeling**: Utilizes Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) to identify and describe topics in movie descriptions.
- **Entity Extraction**: Employs Named Entity Recognition (NER) to extract important entities from movie descriptions.
- **Keyword Extraction**: Uses TF-IDF to identify key terms in movie descriptions.
- **Sentiment Analysis**: Analyzes the sentiment of movie descriptions to gain insights into their tone and mood.
- **Similarity Calculation**: Computes cosine similarity between selected movies and other movies to provide personalized recommendations.

## Data

- **Source**: The dataset is assumed to be a CSV file located at `/Users/danielbrown/Desktop/data.csv`.
- **Columns**: Includes columns such as `Title`, `Description`, `Year`, `Director`, `Country`, `Language`, and `Image`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/djbrown227/criterion_collection_recommendations.git
