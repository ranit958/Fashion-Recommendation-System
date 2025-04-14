import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
from tensorflow.keras.models import load_model

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
model = load_model("model.h5")

img = image.load_img('sample/18714.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbours.fit(feature_list)

distances, indices = neighbours.kneighbors([normalized_result])

for file in indices[0]:
    temp_img = cv.imread(filenames[file])
    cv.imshow('output', cv.resize(temp_img, (512,512)))
    cv.waitKey(0)



