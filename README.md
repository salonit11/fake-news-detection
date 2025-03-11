# Fake News Detection

## Project Overview
This project focuses on detecting fake news using machine learning models. We preprocess the dataset, apply TF-IDF vectorization on the text column, and train multiple classification models. The best-performing model is fine-tuned using RandomizedSearchCV.

## Dataset
- The dataset contains news articles labeled as **Fake** or **Real**.
- Only the **text** column is used for feature extraction.
- Labels are mapped as:
  - **Fake** → 0
  - **Real** → 1

## Data Preprocessing
1. **Cleaning the Text Data**
   - Removed special characters and extra spaces.
   - Converted text to lowercase.
   - Removed stopwords.
2. **Feature Extraction using TF-IDF Vectorization**
   - Extracted important words using Term Frequency-Inverse Document Frequency (TF-IDF).
   - Used **max_features=5000** to limit vocabulary size.

## Machine Learning Models Used
We trained the following classifiers to compare performance:
1. **Decision Tree Classifier**
2. **Logistic Regression**
3. **Random Forest Classifier**
4. **KNeighborsClassifier**
5. **Support Vector Machine (SVM)**

## Hyperparameter Tuning
- The best-performing model was **Random Forest Classifier**.
- We used **RandomizedSearchCV** to tune hyperparameters:
  - `n_estimators`: [50, 100, 200, 500]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `bootstrap`: [True, False]

## Model Evaluation
- Models were evaluated using **Accuracy, Classification Report**.
- The best accuracy was achieved by **Random Forest**.

## Installation & Usage
### Requirements
Kaggle Dataset link: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv
```bash
pip install numpy pandas scikit-learn
```

### Run the Project
```bash
python fake_news_detection.py
```

## Results
| Model                  | Accuracy |
|------------------------|----------|
| Decision Tree          | 85.2%    |
| Logistic Regression    | 92.5%    |
| Random Forest         | **96.3%**  |
| KNeighborsClassifier   | 88.7%    |
| SVM                    | 91.8%    |

## Conclusion
- **Random Forest performed the best** in detecting fake news.
- **TF-IDF vectorization** helped in extracting meaningful features.
- **Hyperparameter tuning improved accuracy.**

## Future Improvements
- Experimenting with deep learning models such as LSTMs or Transformers.
- Using additional metadata like article source and author for better accuracy.

---
🚀 *Happy Coding!*

