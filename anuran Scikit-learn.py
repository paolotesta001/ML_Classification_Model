# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:11:47 2025

@author: Utente
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import seaborn as sns

dataset = pd.read_csv('anuran_calls.csv')

# Mapping dictionaries
family_mapping = {
    'Bufonidae': 1,
    'Dendrobatidae': 2,
    'Hylidae': 3,
    'Leptodactylidae': 4,
}
genus_mapping = {
    'Adenomera': 1,
    'Ameerega': 2,
    'Dendropsophus': 3,
    'Hypsiboas': 4,
    'Leptodactylus': 5,
    'Osteocephalus': 6,
    'Rhinella': 7,
    'Scinax': 8,
}
species_mapping = {
    'AdenomeraAndre': 0,
    'AdenomeraHylaedactylus': 1,
    'Ameeregatrivittata': 2,
    'HylaMinuta': 3,
    'HypsiboasCinerascens': 4,
    'HypsiboasCordobae': 5,
    'LeptodactylusFuscus': 6,
    'OsteocephalusOophagus': 7,
    'Rhinellagranulosa': 8,
    'ScinaxRuber': 9,
}

# Apply the mapping
dataset_mod = dataset.copy()
dataset_mod['Family'] = dataset_mod['Family'].map(family_mapping)
dataset_mod['Genus'] = dataset_mod['Genus'].map(genus_mapping)
dataset_mod['Species'] = dataset_mod['Species'].map(species_mapping)

X = dataset_mod.drop('Species', axis=1).values
y = dataset_mod['Species'].values

# Split 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Logistic Regression model
clf = LogisticRegression(
    solver='lbfgs',             # Recommended for multinomial
    max_iter=5000,              # Number of epochs (iterations)
    C=1e3,                      # Inverse of regularization, very high = little regularization
    verbose=1                   # To see convergence (optional)
)
clf.fit(X_train, y_train)

n_epochs = 5000
train_losses = []
val_losses = []

clf = SGDClassifier(loss='log_loss', max_iter=50000, learning_rate='constant', eta0=0.01, warm_start=True, random_state=42)

for epoch in range(n_epochs):
    clf.fit(X_train, y_train)
    y_train_proba = clf.predict_proba(X_train)
    y_val_proba = clf.predict_proba(X_val)
    train_losses.append(log_loss(y_train, y_train_proba))
    val_losses.append(log_loss(y_val, y_val_proba))

plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.show()

# Accuracy
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

print(f"Training set accuracy: {accuracy_score(y_train, y_train_pred)*100:.2f}%")
print(f"Validation set accuracy: {accuracy_score(y_val, y_val_pred)*100:.2f}%")
print(f"Test set accuracy: {accuracy_score(y_test, y_test_pred)*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Precision, Recall, F1-score, Specificity
report = classification_report(y_test, y_test_pred, target_names=species_mapping.keys())
print("Classification Report:\n", report)

# Specificity (for each class: TN / (TN + FP))
def specificity_score(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_per_class = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        fp = cm[:,i].sum() - cm[i,i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    return specificity_per_class

specs = specificity_score(y_test, y_test_pred, labels=list(range(len(species_mapping))))
for i, s in enumerate(specs):
    print(f"Specificity for class {list(species_mapping.keys())[i]}: {s:.2f}")

# ROC curve and AUC (one vs all)
y_test_bin = np.eye(len(species_mapping))[y_test]  # One-hot encoding
y_score = clf.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(species_mapping)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,8))
for i in range(len(species_mapping)):
    plt.plot(fpr[i], tpr[i], label=f'ROC class {list(species_mapping.keys())[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve (One-vs-All)')
plt.legend()
plt.show()

# Macro-average AUC
macro_auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
print(f"Macro-average AUC: {macro_auc:.2f}")
