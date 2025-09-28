import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

model = load_model("saved_models/best_model.h5")
X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = np.mean(y_pred_classes == y_true)
print("Test Accuracy:", acc)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Biodegradable", "Recyclable", "Non-Recyclable"])
disp.plot()
plt.show()
