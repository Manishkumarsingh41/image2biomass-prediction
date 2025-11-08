import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Load and preprocess the dataset
# Replace 'path/to/dataset.csv' with the actual path to your dataset

# Define the paths for images and targets
image_path = 'path/to/images/'
dataset_path = 'path/to/dataset.csv'

# Load dataset
data = pd.read_csv(dataset_path)
X = data['image_path'].apply(lambda x: image_path + x).values
Y = data[['target1', 'target2', 'target3', 'target4', 'target5']].values

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5))  # Output layer for 5 target biomass

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

# Define custom R² scoring function
def weighted_r2_score(y_true, y_pred):
    weights = np.ones_like(y_true)  # Update weights here if needed
    return r2_score(y_true, y_pred, sample_weight=weights)

# Wrap custom scoring function
r2_scorer = make_scorer(weighted_r2_score)

# Train the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)

# Predict on validation set
Y_pred = model.predict(X_val)

# Model performance evaluation
print('R² Score:', weighted_r2_score(Y_val, Y_pred))

# Create submission
submission = pd.DataFrame(data=Y_pred, columns=['target1', 'target2', 'target3', 'target4', 'target5'])
submission.to_csv('submission.csv', index=False)