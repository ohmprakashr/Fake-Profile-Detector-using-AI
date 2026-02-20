# coding: utf-8

### Detect fake profiles in online social networks using Random Forest
 
import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer  # Updated import
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve  # Updated import
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib  # For saving the model

# Function for reading datasets from csv files
def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    x = pd.concat([genuine_users, fake_users])   
    y = len(fake_users) * [0] + len(genuine_users) * [1]
    return x, y

# Function for predicting sex using name of person (removed sexmachine dependency)
def predict_sex(name):
    # For now, a simple approach for sex prediction based on name length or first letter
    sex = name.apply(lambda x: 1 if len(x.split()[0]) % 2 == 0 else 0)  # Simple heuristic
    return sex

# Function for feature engineering
def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = {name: i for i, name in lang_list}             
    x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)    
    x.loc[:, 'sex_code'] = predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    x = x.loc[:, feature_columns_to_use]
    return x

# Function for plotting learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Function for plotting confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['Fake', 'Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Function for plotting ROC curve
def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    print("False Positive rate: ", false_positive_rate)
    print("True Positive rate: ", true_positive_rate)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Function for training data using Random Forest
def train(X_train, y_train, X_test):
    """ Trains and predicts dataset with a Random Forest classifier """
    
    clf = RandomForestClassifier(n_estimators=40, oob_score=True)
    clf.fit(X_train, y_train)
    print("The best classifier is: ", clf)
    # Estimate score
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    title = 'Learning Curves (Random Forest)'
    plot_learning_curve(clf, title, X_train, y_train, cv=5)
    plt.show()
    # Predict 
    y_pred = clf.predict(X_test)
    
    # Save the trained model
    model_filename = 'random_forest_model.pkl'
    joblib.dump(clf, model_filename)
    print(f"Model saved as {model_filename}")
    
    return y_test, y_pred

# Main code
print("Reading datasets.....\n")
x, y = read_datasets()
print(x.describe())

print("Extracting features.....\n")
x = extract_features(x)
print(x.columns)
print(x.describe())

print("Splitting datasets into train and test dataset...\n")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

print("Training datasets.......\n")
y_test, y_pred = train(X_train, y_train, X_test)

print('Classification Accuracy on Test dataset: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

plot_roc_curve(y_test, y_pred)
