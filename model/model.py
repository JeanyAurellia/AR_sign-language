from tensorflow.keras.models import load_model
model = load_model('sign_model.h5')
print(model.summary())