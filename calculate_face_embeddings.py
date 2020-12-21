# Packages for creating face embeddings
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# Get the face embedding for one face based on a specified deep network model

def get_embedding(model, face_pixels):
    
    face_pixels = face_pixels.astype('float32') # scale pixel values

    
    # Pre-whitening process: pixel values across channels (globally)
    mean, std = face_pixels.mean(), face_pixels.std() 
    face_pixels = (face_pixels - mean) / std
    
    # (OPTIONAL) L2-normalisation: enhance facial characteristics
    
    samples = expand_dims(face_pixels, axis = 0)
    
    
    yhat = model.predict(samples)
    
    return yhat[0]

