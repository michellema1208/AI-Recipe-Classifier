import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

loaded = np.load("raw_data.npz")
testIngredients = loaded["testIngredients"]
testCuisine = loaded["testCuisine"]
allIngredients = loaded["allIngredients"]

#### TEST ####
boolTestIngredients = np.zeros((len(testIngredients), len(allIngredients)))
for i in range(len(testIngredients)):
    recipeIngredients = testIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTestIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolTestCuisine = vectorizer.fit_transform(testCuisine).toarray()
boolTestCuisine = boolTestCuisine.argmax(1)

np.savez_compressed("test", boolTestIngredients = boolTestIngredients, boolTestCuisine = boolTestCuisine, dtype = bool)
