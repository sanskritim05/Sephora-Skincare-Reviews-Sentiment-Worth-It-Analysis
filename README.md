# Sephora Skincare Reviews: Sentiment & Worth It Analysis

This project analyzes Sephora skincare reviews to rank products and brands using a custom **Worth It Score**.  
Instead of relying only on star ratings, it uses review text sentiment, review volume, and price information to better capture which products are actually worth buying.

The project has two main parts:
- A **Jupyter notebook** that trains a sentiment model and computes product scores
- A **Streamlit app** that displays the results interactively

---

## Project Overview

Skincare reviews can be misleading:
- Products with very few reviews can appear unrealistically good
- Star ratings alone do not capture how people describe their experience
- Higher price does not always mean higher quality

This project addresses these issues by:
- Predicting sentiment from review text
- Applying Bayesian smoothing to reduce small-sample bias
- Including a simple price-based value component
- Allowing users to explore results by product and by brand

---

## What the notebook does  
**`Sephora_sentiment_worth_it.ipynb`**

The notebook builds everything needed for the app.

### 1. Load and merge data
- Loads all review files from `data/reviews_*.csv`
- Loads product metadata from `data/product_info.csv`
- Merges reviews with products using `product_id`

### 2. Clean review text
- Lowercases text
- Removes HTML, URLs, and special characters
- Normalizes whitespace

### 3. Train a sentiment model
- Uses star ratings as weak sentiment labels:
  - 1–2 → negative  
  - 3 → neutral  
  - 4–5 → positive
- Model: **TF-IDF + Logistic Regression**
- Uses a stratified train/test split for evaluation

### 4. Predict sentiment for all reviews
- Predicts sentiment for every review
- Creates a binary `is_positive` indicator

### 5. Aggregate to the product level
For each product, the notebook computes:
- Number of reviews
- Average rating
- Positive review rate
- Average price (if available)

### 6. Apply Bayesian smoothing
- Prevents products with few reviews from ranking unrealistically high or low
- Adjusts ratings and positive rates toward global averages

### 7. Compute the Worth It Score
The final score combines:
- Bayesian-adjusted rating
- Bayesian-adjusted positive sentiment rate
- A price-based value score

### 8. Save outputs
The notebook saves:
- `outputs/product_rankings.csv`
- `models/sentiment_model.joblib`

---

## What the app does  
**`app.py`**

The Streamlit app reads `outputs/product_rankings.csv` and displays the results.

### Products tab
- Filter by brand, minimum reviews, and product name
- Sort by worth it score, rating, review count, or price
- View:
  - Summary statistics
  - Top 10 products chart
  - Product table with CSV download
  - Detailed product score breakdown

### Brands tab
- Ranks brands using review-weighted averages across products
- View:
  - Top brands chart
  - Brand table with CSV download
  - Brand details and top products for each brand

---

## How to run the project

### 1) Clone the repository
```bash
git clone https://github.com/sanskritim05/Sephora-Skincare-Reviews-Sentiment.git
cd Sephora-Skincare-Reviews-Sentiment
```

### 2) Put the dataset in the right folders
Your project should include:
``` bash
data/
├── product_info.csv
├── reviews_0-25000.csv
├── reviews_25000-50000.csv
└── reviews_*.csv
```
(The review files may be split into multiple chunks depending on the dataset.)

### 3) Install dependencies
Install all required packages using requirements.txt:
pip install -r requirements.txt

### 4) Run the notebook (creates the files the app needs)
Open and run:
sephora_sentiment_worth_it.ipynb

This will generate:
outputs/
└── product_rankings.csv

models/
└── sentiment_model.joblib

### 5) Run the Streamlit app
streamlit run app.py

## Project structure
.
├── app.py
├── sephora_sentiment_worth_it.ipynb
├── data/
│   ├── product_info.csv
│   └── reviews_*.csv
├── models/          # generated
│   └── sentiment_model.joblib
├── outputs/         # generated
│   └── product_rankings.csv
├── requirements.txt
├── .gitignore
└── README.md

## Notes
- Rankings reflect overall review trends and do not account for individual skin type or sensitivities.
- This project is for educational and analytical purposes only. It is not medical or dermatological advice. 
