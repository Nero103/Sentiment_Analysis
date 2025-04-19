# Elemental Movie Review Sentiment Analysis

## Project Overview
This project analyzes audience sentiment for Pixar's Elemental using real user reviews from IMDb and Metacritic. It explores how people reacted to the film and whether review text alone can predict whether a review is positive or negative. A logistic regression model was built for sentiment classification, and an interactive dashboard visualizes key trends.

This is an end-to-end project on sentiment analysis, covering data collection, preprocessing, analysis, machine learning, and dashboard visualization.

## Objectives
* Analyze user review data to understand viewer sentiment

* Explore which types of reviews are considered helpful

* Visualize patterns in ratings and review timing

* Build a machine learning model that classifies sentiment from review text

## Data Collection & Preprocessing
Source: Data was collected using Listly (a Chrome extension) from IMDb and Metacritic

Initial Work: Four datasets were cleaned, formatted, and merged using Microsoft Excel Power Query

Cleaning Steps:

1. Removed duplicates

2. Normalized rating scales to a 1–5 scale called NewRating

2. Handled missing values

Workflow:

After preprocessing in Power Query, the combined dataset was uploaded into Jupyter Notebook for analysis and machine learning

Three final CSVs were exported for use in Tableau to build the visual dashboard

## Exploratory Data Analysis (EDA)
Ratings:

* Most ratings clustered around 4 and 5, indicating generally positive sentiment

Helpfulness Votes:

* Very few reviews received helpful votes

* Among helpful reviews, those rated 2 were surprisingly dominant—possibly due to longer, more detailed critiques or perceived honesty

Wordcloud:

* Common words included "love", "family", "story", and "beautiful"—suggesting positive emotional resonance

Time of Reviews:

* Review volume peaked near the movie's release date (June 16, 2023), showing strong early interest

## Dashboard Insights

![Dashboard 1 (2)](https://github.com/user-attachments/assets/fa0c46a0-7922-48a8-a3ee-c6ddc0010e63)

The interactive dashboard in Tableau provides key insights, following a Z-pattern:

**Sentiment Column Chart**: a column chart showing the distribution of sentiment scores from June to July, broken down by sentiment category, shows June had more ratings than July , yet negative views dropped significantly in July. The timeline captures the burst in review activity close to release

**Wordcloud**: Highlights common sentiment-heavy words used in reviews

**Review Distribution**: A bar chart shows the total number of reviews fell into each NewRating based on (1–5 rating), broken down by categorized in Positive, Negative, Neutral

**Helpfulness vs. Rating**: Scatter plot shows how helpfulness votes relate to review ratings by reviewer, highlighting that lower-rated reviews were often perceived as more helpful

## Sentiment Analysis Approach
To train a sentiment classifier, reviews were labeled:

* Positive (1) if NewRating > 3

* Negative (-1) if NewRating < 3

* Neutral reviews (rating = 3) were excluded from the model training

## Machine Learning Model
Model Used: **Logistic Regression**

Pipeline:

* Text data vectorized with TfidfVectorizer

* Logistic regression trained to classify sentiment

Accuracy:
* Achieved an accuracy of ~85.5%, performing well in distinguishing positive from negative reviews

Confusion Matrix:
* Showed balanced performance across both classes with minimal misclassification

## Interpretations
**Positive Buzz**: Viewers generally liked the movie, especially emphasizing themes like family and emotional storytelling

**Trust in Negativity**: Lower-rated reviews, particularly 2s, were more likely to receive helpfulness votes—possibly because readers trust critical reviews as being more in-depth or honest

**Model Reliability**: The logistic regression model performed well using only review text, proving that even simple models can yield high accuracy when paired with meaningful features

**Release Momentum**: Review frequency spiked at launch and gradually tapered—showing that audience sentiment is heavily concentrated around a movie’s initial hype window

## Tools & Libraries
Excel Power Query – Initial formatting, cleaning, merging

Pandas, NumPy – Data manipulation

Seaborn, Matplotlib – Visualizations

Scikit-learn – Machine learning & evaluation

Tableau – Final dashboard visualization

Listly – Web scraping tool

## Conclusion
Elemental received generally positive feedback

Negative reviews were often considered more helpful, possibly due to detail or tone

The logistic regression model accurately classified sentiment using only review text

The dashboard provides an intuitive way to explore viewer sentiment, review behavior, and content trends

The project successfully tied together scraping, cleaning, modeling, and visualization into a complete end-to-end data science pipeline
