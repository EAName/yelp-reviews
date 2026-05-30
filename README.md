# yelp-reviews

Graduate capstone project on production Yelp business and review data to identify drivers of low restaurant ratings and actionable levers for 1–3 star businesses to improve customer consideration and traffic.

---

## 1. Title and Summary

**Yelp Reviews Capstone**  
Northwestern University M.S. in Data Science: group capstone analyzing production-scale Yelp business, review, and user data end to end — from JSON ingestion and restaurant-focused cleaning through market-level EDA, supervised rating models, NLP on review text, and unsupervised segmentation — with a St. Louis market deep dive for operational recommendations.

---

## 2. Concepts and Methods

- **Data ingestion:** convert production Yelp JSON exports to CSV for business, review, and user entities (`Yelp_Data_Converting_Review_Json_to_CSV.ipynb`, `Yelp_Data_Converting_User_Json_to_CSV.ipynb`, `Create_Reviews_CSV_Version_2.ipynb`, `Create_Clean_User_and_Review_CSV_Version_4.ipynb`)
- **Business dataset cleaning:** filter to restaurant categories; drop non-U.S. records; harmonize attributes and market fields (`Cleaning_Business_Dataset.ipynb`, `YelpBusinessClean5.ipynb`)
- **Review–business integration:** merge review text with business metadata; tokenize and normalize review bodies for downstream NLP (`Create_Combined_Business_and_Review_File.ipynb`)
- **Exploratory analysis:** state-to-market mapping; category and chain frequency profiling; star-rating distributions; interactive review dashboards (`Exploratory_Data_Analysis.ipynb`, `EDA_yelp_business.ipynb`, `EDA_Yelp_Business_Rabia.ipynb`, `EDA_yelp_business_cleaned_v4.ipynb`, `EDA_yelp_reviews_dashboard_v2.ipynb`)
- **St. Louis market focus:** subset Missouri/St. Louis restaurants; linear regression on predicted star ratings for local benchmarking (`YelpBusiness_St_Louis.ipynb`, `Yelp_business_logistic_stlouis.R`)
- **Supervised tabular models:** linear and logistic regression on business attributes (hours, parking, WiFi, attire, cuisine flags); random forest and XGBoost for star-rating prediction (`yelp_business_linear_regression.ipynb`, `yelp_business_logistic_regression.ipynb`, `Yelp_Linear_Regression_EE.ipynb`, `Yelp_RandomForest_XGBoost_AA.ipynb`, `Yelp_Combined_Models_RA.ipynb`)
- **Dimensionality reduction and clustering:** PCA on business feature matrices; k-means cluster evaluation with `caret`, `factoextra`, and `NbClust` (`Yelp_business_PCA_EE1027.R`, `Cluster_yelp_business_EE.R`, `yelp_business_logistic_regression_R.R`)
- **Review NLP — classical:** punctuation and stopword removal; bag-of-words and TF-IDF vectorization; star-stratified word clouds and top-term summaries (`Yelp_NLP_Bag_of_Words_RA.ipynb`, `Yelp_Top_Words_for_Each_Business_RA.ipynb`, `yelp_review_NLP_maureenNB0.ipynb`)
- **Review NLP — sentiment and topics:** TextBlob/sentiment-style analysis; LDA and NMF topic modeling on review corpora (`Yelp_NLP_SentimentAnalysis_Maureen.ipynb`, `Yelp_NLP_Topic_Modeling.ipynb`, `Yelp_NLP_RA.ipynb`)
- **Review NLP — deep learning:** Keras dense and LSTM/RNN models on text features; combined review-only DNN without business attributes for ablation (`Yelp_NLP_RNN_RA.ipynb`, `Yelp_Combined_Reviews_(no_business_attributes)_DNN_RA.ipynb`, `Yelp_NLP_ABSA_Maureen.ipynb`)
- **Aspect-based sentiment (ABSA):** extract aspect-level signals from review text to connect complaint themes to star tiers (`Yelp_NLP_ABSA_Maureen.ipynb`)

**Business objective:** prioritize operational and service improvements for lower-tier (1–3 star) restaurants by combining attribute-based models with text-derived complaint themes

**Data dependencies:** production Yelp JSON/CSV files are referenced from Google Drive/Colab paths and are not bundled in this repository

---

## 3. Stack

| Layer | Tools |
|-------|-------|
| Languages | Python 3, R |
| Environment | Jupyter Notebook, Google Colab |
| Data | pandas, NumPy, JSON/CSV I/O |
| Classical ML | scikit-learn (linear/logistic regression, random forest), XGBoost |
| NLP | NLTK, regex, CountVectorizer, TF-IDF, WordCloud, TextBlob-style sentiment |
| Topic modeling | LDA, NMF |
| Deep learning | TensorFlow 2 / Keras (dense, LSTM/RNN) |
| R analytics | caret, tidyverse, psych, nnet, pROC, cluster, factoextra, DataExplorer |
| Visualization | matplotlib, seaborn |

---

## 4. Structure

```
yelp-reviews/
├── Yelp_Data_Converting_Review_Json_to_CSV.ipynb
├── Yelp_Data_Converting_User_Json_to_CSV.ipynb
├── Create_Reviews_CSV_Version_2.ipynb
├── Create_Clean_User_and_Review_CSV_Version_4.ipynb
├── Cleaning_Business_Dataset.ipynb
├── Create_Combined_Business_and_Review_File.ipynb
├── Exploratory_Data_Analysis.ipynb
├── EDA_yelp_business.ipynb
├── EDA_Yelp_Business_Rabia.ipynb
├── EDA_yelp_business_cleaned_v4.ipynb
├── EDA_yelp_reviews_dashboard_v2.ipynb
├── YelpBusinessClean5.ipynb
├── YelpBusiness_St_Louis.ipynb
├── yelp_business_linear_regression.ipynb
├── yelp_business_logistic_regression.ipynb
├── Yelp_Linear_Regression_EE.ipynb
├── Yelp_RandomForest_XGBoost_AA.ipynb
├── Yelp_Combined_Models_RA.ipynb
├── yelp_review_NLP_maureenNB0.ipynb
├── Yelp_NLP_Bag_of_Words_RA.ipynb
├── Yelp_NLP_RA.ipynb
├── Yelp_NLP_SentimentAnalysis_Maureen.ipynb
├── Yelp_NLP_Topic_Modeling.ipynb
├── Yelp_NLP_RNN_RA.ipynb
├── Yelp_NLP_ABSA_Maureen.ipynb
├── Yelp_Top_Words_for_Each_Business_RA.ipynb
├── Yelp_Combined_Reviews_(no_business_attributes)_DNN_RA.ipynb
├── Cluster_yelp_business_EE.R
├── Yelp_business_PCA_EE1027.R
├── Yelp_business_logistic_stlouis.R
├── yelp_business_logistic_regression_R.R
└── README.md
```

- **Organization:** ingest/clean → combine → EDA → tabular models → NLP/classical → NLP/deep learning → R-side PCA/clustering; St. Louis notebooks isolate a target market
- **Reusable modules:** none packaged; cleaning helpers and model pipelines defined inline per notebook
- **Engineering practice:** versioned CSV build notebooks (`Version_2`, `Version_4`, `Clean5`); star-tier stratification for text analysis; model comparisons across attribute-only, review-only, and combined feature sets; Colab-ready workflow with GitHub sync

---

**Course context:** Northwestern University, M.S. in Data Science — graduate capstone project  
**Repository:** https://github.com/EAName/yelp-reviews
