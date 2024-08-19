#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[318]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[319]:


# Load the dataset
file_path = '/Users/danielbrown/Desktop/data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()


# In[320]:


#Basic EDA, this data has largely been cleaned


# In[321]:


# Shape of the dataframe
print("Shape of the dataframe:", df.shape)

# Data types of each column
print("\nData types of each column:\n", df.dtypes)

# Summary statistics
print("\nSummary statistics:\n", df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())


# In[322]:


# Convert the 'Year' column to integer type after datetime conversion
df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce').dt.year.astype('Int64')

# Display the first few rows to verify the change
df.head()


# Drop the 'Image' and 'Unnamed: 0' columns
df = df.drop(columns=['Image', 'Unnamed: 0'])

# Display the first few rows of the modified dataframe
df.head()


# In[323]:


#More work can be done here, manually filling in missing data to create a more 
#complete dataset


# In[324]:


# Filter and display rows with any missing values
rows_with_missing_values = df[df.isnull().any(axis=1)]

# Print the rows with missing values
print(rows_with_missing_values)


# In[325]:


# Fill missing 'Country' and 'Language' with 'Unknown'
df['Country'].fillna('Unknown', inplace=True)
df['Language'].fillna('Unknown', inplace=True)

# Drop rows with missing 'Director' or 'Year'
df = df.dropna(subset=['Director', 'Year'])

# Display the modified dataframe
df.head()


# In[326]:


# Shape of the dataframe
print("Shape of the dataframe:", df.shape)


# In[327]:


#Make the description column all lowercase so the algorithms ca habdle the data


# In[328]:


import nltk
from nltk.corpus import stopwords
import string

# Example of text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['Cleaned_Description'] = df['Description'].apply(preprocess_text)


# In[329]:


df.head()


# In[330]:


'''
Topic Modeling is a method to discover abstract topics in a set of documents. 
This can help you categorize the descriptions into different genres or themes. 
Two common algorithms are:

Latent Dirichlet Allocation (LDA): Useful for finding hidden topics in the descriptions.
'''


# In[331]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Example with LDA
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['Cleaned_Description'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# You can then assign the most prominent topic as a new feature for each movie
df['Topic'] = lda.transform(dtm).argmax(axis=1)


# In[332]:


import numpy as np

# Step 1: Extract top words for each topic
def get_top_words(lda_model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(" ".join(top_words))
    return topics

# Get top words for each topic
topic_descriptions = get_top_words(lda, vectorizer)

# Step 2: Map topic numbers to descriptions
topic_mapping = {i: desc for i, desc in enumerate(topic_descriptions)}

# Step 3: Add descriptive topic column
df['Topic_Description'] = df['Topic'].map(topic_mapping)

# Optional: Print the DataFrame to see the results
#print(df[['Title', 'Topic', 'Topic_Description']])


# In[333]:


df.head()


# In[334]:


'''
Named Entity Recognition (NER)
NER helps extract specific entities from text, such as names of people, 
locations, or even genres. You can use libraries like spaCy for this.

This method can help identify entities like film titles, director names, 
and possibly genres or other relevant information.
'''


# In[335]:


import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['Entities'] = df['Description'].apply(extract_entities)


# In[336]:


'''
Keyword Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) is a method to 
extract the most important words in a text. 
These keywords can give you insights into the plot and themes of each movie.
'''


# In[337]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Example of TF-IDF
tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(df['Cleaned_Description'])

keywords = tfidf.get_feature_names_out()
df['Keywords'] = [', '.join([keywords[idx] for idx in tfidf_matrix[row].indices]) for row in range(tfidf_matrix.shape[0])]


# In[338]:


'''
Sentiment Analysis
Analyzing the sentiment of the movie descriptions can also be valuable. It might give 
clues about the tone or mood of the movie, which could influence user preferences.
'''


# In[339]:


'''
from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Cleaned_Description'].apply(get_sentiment)
'''


# In[340]:


'''
Non-Negative Matrix Factorization (NMF): Another algorithm that can be used 
for topic modeling.
'''


# In[341]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# In[342]:


# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit and transform the description column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Description'])


# In[343]:


# Number of topics
n_topics = 5

# Fit the NMF model
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_matrix = nmf_model.fit_transform(tfidf_matrix)


# In[344]:


# Get the topic that has the highest score for each movie
df['Dominant_Topic'] = nmf_matrix.argmax(axis=1)


# In[345]:


# Get the top words for each topic
def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[topic_idx] = top_words
    return topics

# Display the top words for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
top_words_per_topic = get_top_words(nmf_model, feature_names, 10)
print(top_words_per_topic)


# In[346]:


df.head()


# In[347]:


df['Director_Popularity'] = df['Director'].map(df['Director'].value_counts())


# In[348]:


df.head()


# In[349]:


df['Country_Popularity'] = df['Country'].map(df['Country'].value_counts())
df['Language_Popularity'] = df['Language'].map(df['Language'].value_counts())


# In[350]:


df.head()


# In[351]:


df['Decade'] = (df['Year'] // 10) * 10


# In[352]:


df.head()


# In[353]:


def get_user_movies():
    print("Please enter two movies from the Criterion Collection:")
    movie1 = input("Enter the first movie: ")
    movie2 = input("Enter the second movie: ")
    return movie1, movie2


# In[354]:


def find_movies(df, movie1, movie2):
    movie1_data = df[df['Title'].str.contains(movie1, case=False, na=False)]
    movie2_data = df[df['Title'].str.contains(movie2, case=False, na=False)]
    return movie1_data, movie2_data


# In[ ]:





# In[372]:


from sklearn.metrics.pairwise import cosine_similarity

def get_recommendation(df, movie1_data, movie2_data):
    # Combine the features of the two selected movies
    selected_movies_features = pd.concat([movie1_data, movie2_data]).mean(axis=0)
    
    # Select only the numeric columns for similarity calculation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute cosine similarity between selected movies and the rest of the dataset
    similarity_scores = cosine_similarity([selected_movies_features], numeric_df.values)
    
    # Get the highest similarity score that is not one of the selected movies
    df['Similarity'] = similarity_scores[0]
    recommendation = df.loc[~df['Title'].isin([movie1_data['Title'].values[0], movie2_data['Title'].values[0]])].sort_values(by='Similarity', ascending=False).iloc[0]
    
    # Extract relevant information for the recommendation
    recommended_title = recommendation['Title']
    recommended_description = recommendation['Description']
    similarity_score = recommendation['Similarity']
    matching_features = {
        'Topic': recommendation['Topic'],
        'Director': recommendation['Director'],
        'Decade': recommendation['Decade'],
        'Country': recommendation['Country'],
        'Language': recommendation['Language']
    }
    
    # Create a prose explanation
    prose_explanation = (
        #f"Based on your interest in '{movie1_data['Title'].values[0]}' and '{movie2_data['Title'].values[0]}', "
        #f"we've selected '{recommended_title}' as a movie you might enjoy. "
        #f"Like the first movie, '{movie1_data['Title'].values[0]}', this recommendation is set in the {matching_features['Decade']}s and is directed by a renowned filmmaker, {matching_features['Director']}. "
        #f"It shares thematic elements with '{movie2_data['Title'].values[0]}', exploring similar topics such as {matching_features['Topic']}. "
        #f"The language and cultural context of the film also align closely with your preferences, making it a fitting choice for your next watch."
    )
    
    # Prepare the output message
    output_message = (
        f"We recommend you watch: {recommended_title}\n"
        f"Description: {recommended_description}\n\n"
        #f"This recommendation was chosen because it shares the following features with the movies you selected:\n"
        #f"- Topic: {matching_features['Topic']}\n"
        #f"- Director: {matching_features['Director']}\n"
        #f"- Decade: {matching_features['Decade']}\n"
        #f"- Country: {matching_features['Country']}\n"
        #f"- Language: {matching_features['Language']}\n"
        #f"Similarity Score: {similarity_score:.2f}\n\n"
        #f"{prose_explanation}"
    )
    
    return output_message


# In[373]:


recommend_movie(df)


# In[ ]:





# In[ ]:




