import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Loading all features

# Replacing actual CSV outputs or DataFrame creation.
lexical_df = pd.read_csv('lexical_density.csv')
mattr_df = pd.read_csv('mattr.csv') 
syntactic_df = pd.read_csv('syntactic.csv')
entropy_df = pd.read_csv('entropy.csv')

# Step 2: Merge all features on a common index
features_df = lexical_df.merge(mattr_df, on='filename')
features_df = features_df.merge(syntactic_df, on='filename')
features_df = features_df.merge(entropy_df, on='filename')

# Step 3: Add labels manually or from filenames
# If filename contains 'human' it's original (0), 'machine' is translated (1)
def assign_label(filename):
    if 'human' in filename:
        return 0
    else:
        return 1

features_df['label'] = features_df['filename'].apply(assign_label)

# Step 4: Prepare data for training
X = features_df.drop(columns=['filename', 'label'])
y = features_df['label']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save model needed
import joblib
joblib.dump(clf, 'translationese_classifier.pkl')
