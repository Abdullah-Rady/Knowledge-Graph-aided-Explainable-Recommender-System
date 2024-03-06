# Project Title: Explainable Recommender System with Knowledge Graph

## Overview

This project aims to develop an explainable recommender system that leverages user ratings on movies, along with a knowledge graph containing information about various movies. The primary goal is to generate personalized movie recommendations for users and provide transparent explanations for these recommendations using the knowledge graph. The system adopts a post hoc knowledge graph-based explanation generation approach to enhance the interpretability of the recommendation process.

## Problem Definition

Given a dataset with user ratings and movie information, the project focuses on:

1. Developing a recommendation system to identify movies relevant to a user's preferences.
2. Utilizing a knowledge graph to generate detailed explanations for recommendations.
3. Enhancing transparency and interpretability in the recommendation process.

## Objectives

1. Develop a hybrid recommender module combining collaborative filtering and content-based filtering.
2. Construct a knowledge graph using the Movielens dataset and DBpedia dataset.
3. Implement explanation generation for recommendations based on user similarity, movie similarity, and knowledge graph rules.
4. Evaluate the system's effectiveness through a user study, considering reliability, safety, predictability, and user satisfaction.

## Implementation and Results

### 1. Data Collection and Preprocessing

- Utilized Movielens dataset and DBpedia dataset for knowledge graph construction.
- Cleaned and preprocessed data to ensure consistency and accuracy.

### 2. Knowledge Graph Construction

- Constructed a knowledge graph using triplets representing relationships between entities (movies, actors, directors, etc.).

### 3. Hybrid Recommender Module

- Developed a hybrid recommender module incorporating collaborative filtering and content-based filtering.
- Addressed the cold start problem and improved diversity in recommendations.

### 4. Explanation Generation

#### 4.1 User Similarity

- Utilized cosine similarity to measure similarity between the active user and others.
- Provided explanations based on user similarity to help users understand the relevance of recommendations.

#### 4.2 Movie Similarity

- Considered genres of top-rated movies for the active user to determine movie similarity.
- Generated explanations based on matching genres between recommended and interacted movies.

#### 4.3 Rule Extraction

- Defined rules in the knowledge graph to capture relationships between movies
