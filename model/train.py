import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from mobilenet_model import build_model

X_train, X_test, y_train, y_test = joblib.load("data_preprocessing/split_data.pkl")

model = build_model()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('saved_models/best_model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=20,
                    batch_size=32,
                    callbacks=callbacks)
