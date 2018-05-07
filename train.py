import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

loaded = np.load("raw_data.npz")
trainIngredients = loaded["trainIngredients"]
trainCuisine = loaded["trainCuisine"]
allIngredients = loaded["allIngredients"]

#### TRAIN ####
boolTrainIngredients = np.zeros((len(trainIngredients), len(allIngredients)))
for i in range(len(trainIngredients)):
    recipeIngredients = trainIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTrainIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolTrainCuisine = vectorizer.fit_transform(trainCuisine).toarray()
boolTrainCuisine = boolTrainCuisine.argmax(1)

np.savez_compressed("train", boolTrainIngredients = boolTrainIngredients, boolTrainCuisine = boolTrainCuisine, dtype = bool)
