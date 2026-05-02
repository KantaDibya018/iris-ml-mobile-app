import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. LOAD THE DATA
data = load_iris()
X = data.data
y = data.target

# Split data: 80% Training, 20% Testing (For Verification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TRAINING SECTION
print("="*50)
print("🚀 TRAINING STARTED - IRIS INTELLIGENCE SYSTEM")
print("="*50)

# Random Forest
rf_model = RandomForestClassifier().fit(X_train, y_train)
# Decision Tree
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
# k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# ANN (Neural Network)
y_train_ann = pd.get_dummies(y_train)
ann_model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train_ann, epochs=100, verbose=0)

# 3. VERIFICATION & CONSOLE OUTPUT
def print_metrics(name, model, is_ann=False):
    if is_ann:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    else:
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n✅ MODEL: {name}")
    print(f"📊 ACCURACY: {acc*100:.2f}%")
    print("📋 CONFUSION MATRIX:")
    print(cm)
    print("-" * 30)

# Display everything in Console
print_metrics("Random Forest", rf_model)
print_metrics("Decision Tree", dt_model)
print_metrics("k-Nearest Neighbors", knn_model)
print_metrics("ANN (Neural Network)", ann_model, is_ann=True)

# 4. SAVE THE BRAINS (Original Formats)
joblib.dump(rf_model, 'model_rf.pkl')
joblib.dump(dt_model, 'model_dt.pkl')
joblib.dump(knn_model, 'model_knn.pkl')
ann_model.save('model_ann.h5')

# --- ADDED FOR MOBILE (TFLite Conversion) ---
print("\n📦 Converting ANN model for Mobile (TFLite)...")
converter = tf.lite.TFLiteConverter.from_keras_model(ann_model)
tflite_model = converter.convert()
with open('model_ann.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Mobile version model saved as: model_ann.tflite")
# ---------------------------------------------

print("\n" + "="*50)
print("🎉 ALL MODELS TRAINED AND VERIFIED SUCCESSFULLY!")
print("="*50)