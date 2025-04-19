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

![sentiment_comparison](https://github.com/user-attachments/assets/034c1c66-30b0-4993-a2c8-5bfe7101f6e6)

* Most ratings clustered around 4 and 5, indicating generally positive sentiment

Helpfulness Votes:

* Very few reviews received helpful votes

* Among helpful reviews, those rated 2 were surprisingly dominant—possibly due to longer, more detailed critiques or perceived honesty

Wordcloud:

* Common words included "love", "family", "story", and "beautiful"—suggesting positive emotional resonance

Time of Reviews:

![trend_of_helpful_reviews](https://github.com/user-attachments/assets/7f95ff23-043f-423f-be27-ac120b8cf28d)

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
* Achieved an accuracy of ~87.3%, performing well in distinguishing positive from negative reviews

Evaluation Metrics:
* **Accuracy score**: 0.873
* **Precision score**: 0.951
* **Recall score**: 0.886
* **F1 score**: 0.918

Based on the metrics above, the model got an accuracy of 0.873, or found to be 87.3% accurate.

Given that we are trying to solve a classification problem, accuracy is good to know, especially with the given result. However, this metric has its limitations in our case. Since or data is kept imbalanced, to be representative of the population in our dataset, the model could have difficulty predicting false positives and negatives. This caveat prompts more focus on the precision and recall scores.

Precision and recall will be useful at evaluating the correct predictive capability of our model because both balance false positives and false negatives.

The model displays a precision score of 0.951, suggesting the model is good at predicting true positives - meaning the review was positive - while balancing false positives. Alternatively, the recall score is 0.886, showing a lower performance in predicting true negative - where the review was negatives - while balancing for false negatives. These both give a better understanding of model performance.

The **F1 score** for our model is 0.918. this score balances the precision and recall performance to assess how well the model delivers predictions. Given the results, the F1 score suggests a good predictive power in this model.

Confusion Matrix:

![confusion_matrix_logistic_regression](https://github.com/user-attachments/assets/0e6808f7-d199-4564-9716-09e8287595d8)

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
