import pandas as pd
df = pd.read_csv(r'/Users/jayveenpatel/Downloads/Cancer_Data - Cancer_Data.csv')
print(df.head(20))

columns_to_drop = ['id', 'Unnamed: 32']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(df.head(5))

# handle the categorical values
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
print(df)

X = df.drop(columns=['diagnosis'])  # input features
y = df['diagnosis']  # target
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# all different ML models concepts ------> supervised nd  unsupervised
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load sample dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(y_pred)

# Evaluate the model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")

# heatmap for above data
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix,annot=True,fmt='d')
plt.show()

# predict some values
sample_inputs = X_test[:5]
predict_probs = model.predict_proba(sample_inputs)
predict_classes = model.predict(sample_inputs)
print(predict_probs)
print(predict_classes)

predictions_df = pd.DataFrame(predict_probs, columns=['Prob_Benign (0)', 'Prob_Malignant (1)'])
predictions_df['Predicted Class'] = predict_classes
predictions_df['Actual Class'] = y_test[:5]  # <-- fixed this line
print(predictions_df)

# Decision Tree Classifier()
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)

# accuracy for model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


# Random Forest Classifier() model ---> ensemble learning
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
print(y_pred)

param = {
    'n_estimators':[200,100,300],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['auto','sqrt'],
    'bootstrap':[True,False],
    'random_state':[42]
}

accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")

# SVM --Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear')
)

param_grid = {
    'svc__C':[0.1,1,10],
    'svc__kernel':['linear','rbf'],
    'svc__gamma':['scale',0.01,0.001]
}

grid = GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',verbose=0)
grid.fit(X_train,y_train)


y_pred = grid.predict(X_test)
print(y_pred)

print(grid.best_params_)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=grid.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# Accuracy Visulaization
acc = accuracy_score(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.barplot(x=['Accuracy'],y=[acc])
plt.ylim(0,1)
plt.ylabel('Accuracy Score')
plt.title('Accuracy Visualization')
plt.show()


# Step 1: Separate features and target
X = df.drop(columns='diagnosis')
y = df['diagnosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
svm = SVC(kernel='poly')
svm.fit(X_train_pca,y_train)

import numpy as np
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Support Vector Machine (SVM) Decision Boundary')
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Adaboost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
ada = AdaBoostClassifier(n_estimators=100,random_state=42)
ada.fit(X_train,y_train)
y_pred = ada.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))

# Define models
from sklearn.ensemble import HistGradientBoostingClassifier

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    "Stacking": StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('hgb', HistGradientBoostingClassifier(max_iter=100, random_state=42))
        ],
        final_estimator=LogisticRegression()
    )
}

# Evaluate all models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")

# Display as DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("\nModel Comparison:")
print(results_df)


# Plotting accuracy comparison from results_df
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.ylim(0, 1)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# dump the ML models using 'pickle' and 'joblib'

import pickle
# save model
with open('xgb_model.pkl','wb') as f:
  pickle.dump(model,f)

with open('xgb_model.pkl','rb') as f:
  model = pickle.load(f)

import joblib

joblib.dump(model, 'xgb_model.joblib')
model = joblib.load('xgb_model.joblib')