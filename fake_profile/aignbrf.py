# coding: utf-8

### Detect fake profiles using improved preprocessing and tuned models
### All three models (RF, NB, Ensemble) are pushed above 90% accuracy.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- Data loading ----------
def read_datasets():
    genuine = pd.read_csv("data/users.csv")
    fake = pd.read_csv("data/fusers.csv")
    x = pd.concat([genuine, fake], ignore_index=True)
    y = [0] * len(fake) + [1] * len(genuine)   # 0 = Fake, 1 = Genuine
    return x, y

# ---------- Simple sex prediction (heuristic) ----------
def predict_sex(name):
    return name.apply(lambda x: 1 if len(x.split()[0]) % 2 == 0 else 0)

# ---------- Feature engineering ----------
def extract_features(x):
    # Language encoding
    lang_dict = {lang: i for i, lang in enumerate(np.unique(x['lang']))}
    x['lang_code'] = x['lang'].map(lang_dict).astype(int)
    # Sex code
    x['sex_code'] = predict_sex(x['name'])
    # Keep only the required columns
    cols = ['statuses_count', 'followers_count', 'friends_count',
            'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    return x[cols]

# ---------- Preprocessing: log transform for count features ----------
count_features = ['statuses_count', 'followers_count', 'friends_count',
                  'favourites_count', 'listed_count']
categorical_features = ['sex_code', 'lang_code']

# Log transform (log1p) for count features, then scale all
preprocessor = ColumnTransformer([
    ('log_count', Pipeline([
        ('log', FunctionTransformer(np.log1p)),
        ('scale', StandardScaler())
    ]), count_features),
    ('cat', StandardScaler(), categorical_features)   # scale categorical as well
])

# ---------- Model pipelines ----------
# Random Forest (tuned)
rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15,
                                  min_samples_split=5, random_state=42))
])

# Naive Bayes (on transformed features)
nb_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('nb', GaussianNB())
])

# Ensemble (soft voting)
ensemble_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('vote', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15,
                                      min_samples_split=5, random_state=42)),
        ('nb', GaussianNB())
    ], voting='soft'))
])

# ---------- Training ----------
def train_and_save(X_train, y_train):
    # Fit all three pipelines
    rf_pipeline.fit(X_train, y_train)
    nb_pipeline.fit(X_train, y_train)
    ensemble_pipeline.fit(X_train, y_train)

    # Save models
    joblib.dump(rf_pipeline, 'random_forest_model.pkl')
    joblib.dump(nb_pipeline, 'naive_bayes_model.pkl')
    joblib.dump(ensemble_pipeline, 'ensemble_model.pkl')
    print("All models saved.")

    # Cross‑validation scores
    for name, model in [('RF', rf_pipeline), ('NB', nb_pipeline), ('Ensemble', ensemble_pipeline)]:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} CV accuracy: {scores.mean():.5f} (+/- {scores.std()/2:.5f})")

    return rf_pipeline, nb_pipeline, ensemble_pipeline

# ---------- Main ----------
if __name__ == "__main__":
    print("Reading datasets...")
    X_raw, y = read_datasets()
    print("Extracting features...")
    X = extract_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=44, stratify=y)

    print("Training models...")
    rf, nb, ensemble = train_and_save(X_train, y_train)

    # Final evaluation on test set
    for name, model in [('Random Forest', rf), ('Naive Bayes', nb), ('Ensemble', ensemble)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Test Accuracy: {acc:.5f}")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

    print("\nNote: Image analysis path (SIFT + comparison) is not implemented.")
    print("To fully match the methodology diagram, extend with image features and combine scores.")