"""
Movie Review Sentiment Analysis

An end-to-end project on the movie Elemental using wordclouds,
visualizations, and a machine learning model on text data.

Dataset: elemental_movie_reviews.xlsx
Sheet: UpdatedMergedReviews2.0
"""

# =========================
# Imports
# ======================

import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

from cleantext import clean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# =========================
# Load Data
# =========================

# Read in dataset through pandas dataframe and assign dataset to variable called "df"
df = pd.read_excel('elemental_movie_reviews.xlsx', sheet_name= 'UpdatedMergedReviews2.0')

# Look at the first 5 rows
df.head()

# Basic statistics of the dataset
df.describe(include = 'all', datetime_is_numeric = True)

# Check the data for missing values
df.isna().sum()

# Locate the row(s) with missing values
missing_data = df.loc[df['NewRating'].isna()]

# show row(s) with missing values
missing_data

# Shape of data before dropping row(s)
df.shape

# Drop the row with missing values
df = df.dropna(axis = 0)

# Confirm row in question with missing data was removed
df.shape

# Validate the correct row of miising values was removed
df.loc[df['NewRating'].isna()]

# Check dataset for duplicates
df.duplicated().sum()

# =========================
# Exploratory Data Analysis
# =========================

# Create histogram of new ratings
fig = sns.histplot(data = df, x = 'NewRating', 
                   binwidth = 1.0,
                  color = 'turquoise')
fig.set_title('Rating Histogram')
plt.show()

# Count the occurrence of each new rating
df['NewRating'].value_counts()

# Show the distribution of review helpful
fig = sns.histplot(data = df, x = 'ReviewHelpful')
fig.set_title('Reviews Helpful Histogram')
plt.show()

# Check the review helpful for outliers
sns.boxplot(data = df, x = 'ReviewHelpful')

# Check the rows with Review Helpful over 11
outlier_df = df.loc[df['ReviewHelpful'] > 11]
outlier_df.head()

# Group the data by rating and get the average ReviewHelpful
grouped_oultlier = outlier_df.groupby(by= 'NewRating').mean()['ReviewHelpful']

# Assign grouped outlier to a dataframe
pd.DataFrame(data = grouped_oultlier)

# Show the trend of Review Helpful
sns.lineplot(x = 'ReviewDate', y = 'ReviewHelpful', data = df, ci = None)
plt.xlabel('Date')
plt.ylabel('Reviews Found Helpful')
plt.xticks(rotation = 90)
plt.show()

# Check reviews with high helpful votes
df.loc[df['ReviewHelpful'] >= 64]

# Filling in missing dates to get a more aligned value to date chart
df2 = df.set_index('ReviewDate').resample('D').first().reset_index()

# Show the trend of Review Helpful
sns.lineplot(x = 'ReviewDate', y = 'ReviewHelpful', data = df2, ci = None)
plt.xlabel('Date')
plt.ylabel('Reviews Found Helpful')
plt.xticks(rotation = 90)
plt.show()

# =========================
# Wordcloud Analysis
# =========================

# Create stopwords list
stopwords = set(STOPWORDS)

# Update stopwords with other common words in the dataset
stopwords.update(['Elemental', 'element', 'elements', 'film', 'movie', 'Pixar', 'character', 'story', 
                  'Ember', 'Wade', 'Disney', 'characters', 'fire', 'water', 'really', 'animation', 'time',
                 'people', 'feel', 'people', 'one'])

# Separate the words
words_text = ' '.join(word for word in df['Review'])

# Create the worldcloud
wordcloud = WordCloud(stopwords = stopwords).generate(words_text)

# Display the plot
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.savefig('wordcloud1.png')
plt.show()

# =========================
# Sentiment Labeling
# =========================

# Assign reviews with a rating > 3 positive sentiment
# rating < 3 negative sentiment
# remove score equals 3
sentiment_df = df.copy()
sentiment_df = sentiment_df.loc[df['NewRating'] != 3]
sentiment_df['sentiment'] = sentiment_df['NewRating'].apply(lambda rating: +1 if rating > 3 else -1)

# Preview the df subset sentiment 
sentiment_df.head()

# Split sentiment df into positive and negative
positive = sentiment_df[sentiment_df['sentiment'] == 1]
negative = sentiment_df[sentiment_df['sentiment'] == -1]

# =========================
# Positive Wordcloud
# =========================

# Positive review words
stopwords = set(STOPWORDS)
stopwords.update(['Elemental', 'element', 'elements', 'film', 'movie', 'Pixar', 'character', 'story', 'good', 
                  'Ember', 'Wade', 'one', 'characters', 'fire', 'water', 'really', 'Disney', 'animation','movies',
                 'love', 'time', 'feel', 'kid', 'kids', 'plot', 'people', 'make', 'world', 'way', 'much', 'well',
                 'thing', 'city', 'will', 'see', 'even', 'films', 'animated'])
pos = ' '.join(word for word in positive['Review'])
wordcloud2 = WordCloud(stopwords = stopwords).generate(pos)

plt.imshow(wordcloud2, interpolation= 'bilinear')
plt.axis('off')
plt.savefig('wordcloud2.png')
plt.show()

# =========================
# Negative Wordcloud
# =========================
stopwords = set(STOPWORDS)
stopwords.update(['Elemental', 'element', 'elements', 'film', 'movie', 'Pixar', 'character', 'story', 'good',
                  'Ember', 'Wade', 'one', 'characters', 'fire', 'water', 'really', 'Disney', 'animation', 'movies',
                 'love', 'thing', 'think', 'time', 'feel', 'kid', 'kids', 'plot', 'people', 'make', 'world', 'another',
                 'overall', 'scene', 'scenes', 'end', 'made', 'first', 'go', 'way', 'making', 'idea', 'see', 'seen', 
                  'things', 'films'])
neg = ' '.join(word for word in negative['Review'])
wordcloud3 = WordCloud(stopwords = stopwords).generate(neg)

plt.imshow(wordcloud3, interpolation= 'bilinear')
plt.axis('off')
plt.savefig('wordcloud2.png')
plt.show()

# =========================
# Sentiment Distribution
# =========================

sentiment_df2 = sentiment_df.copy()
sentiment_df2['sentiment'] = sentiment_df2['sentiment'].replace(-1, 'negative')
sentiment_df2['sentiment'] = sentiment_df2['sentiment'].replace(1, 'positive')

fig= sns.histplot(data = sentiment_df2, x = 'sentiment', color = 'cyan')
fig.set_title('Sentiment Comparison')
plt.show()

# Extract the ratio of positive to negative sentiment
sentiment_df['sentiment'].value_counts(normalize= True)*100

# =========================
# Text Cleaning
# =========================

# Test code for text cleaning
test_str = "I'm here. Is this the fil/m? How is It that this is one of the few Disney/p 🤔"

# Remove puntuations
test_str = test_str.translate(str.maketrans('', '', string.punctuation))

# Remove emojis
test_str = clean(test_str, no_emoji= True)

# View cleaned text
test_str

# Join the words in the text to remove remainder spaces
test_str = ''.join(w for w in test_str)

test_str

# Create a function to clean the review text
def punctuation_remover(text):
    '''
    This function removes puntuation, symbols and emojis from string.
    Translate removes puntuations while clean removes emojis.
    The cleaned text is then assigned to the final_text variable, where the strings are then joined
    This function returns final_text
    '''
    new_text = text
    new_text = new_text.translate(str.maketrans('', '', string.punctuation))
    new_text = clean(text, no_emoji= True)
    final_text = new_text
    
    final_text = ''.join(w for w in text if w not in '/')
    
    return final_text


# Apply the punctuation_remover to the 'Review' column
sentiment_df['Review'] = sentiment_df['Review'].apply(punctuation_remover)


# =========================
# Prepare Modeling Dataset
# =========================

# Create a dataframe with the necessary variables
new_df = sentiment_df[['Review', 'sentiment']]

# Remove reviews of 'No comment'
new_df = new_df.loc[new_df['Review'] != 'No comment']

# Preview dataset
new_df.head()

# Define the y (target) variable
y = new_df['sentiment']

# Define the X (predictor) variable
X = new_df['Review']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25,
                                                   stratify= y, random_state= 42)

# =========================
# Bag of Words
# =========================

# Cerate vectorizer
vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# =========================
# Logistic Regression
# =========================

# Assign logistic regression
lr = LogisticRegression()

# View the shapes of the predictor train and test data to ensure the columns are balanced
print('X training data shape:',X_train.shape)
print('X test data shape:', X_test.shape)

# Fit the data to our logistic regression
lr.fit(X_train, y_train)

# Test our model
predictions = lr.predict(X_test)

# =========================
# Model Evaluation
# =========================

# Display evaluation metrics
print('Accuracy score:', '%.3f' % metrics.accuracy_score(predictions, y_test))
print('Precision score:', '%.3f' % metrics.precision_score(predictions, y_test))
print('Recall score:', '%.3f' % metrics.recall_score(predictions, y_test))
print('F1 score:', '%.3f' % metrics.f1_score(predictions, y_test))

# Construct confusion matrix for the predicted and test values
cm = metrics.confusion_matrix(y_test, predictions, labels = lr.classes_)

# Create display for confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix= cm, display_labels = lr.classes_)

disp.plot()
plt.show()
