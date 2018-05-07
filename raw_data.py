import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

#TRAINING SET
trainData = json.load(open('train.json'))

#parsing data from json file
trainCuisine = []
trainID = []
trainIngredients = []
for i in range(len(trainData)):
    cuisine = trainData[i]["cuisine"]
    trainCuisine.append(cuisine)
    identification = trainData[i]["id"]
    trainID.append(identification)
    ingredients = trainData[i]["ingredients"]
    trainIngredients.append(ingredients)

#convert training set to np array
trainCuisine = np.array(trainCuisine)
trainID = np.array(trainID)
trainIngredients = np.array(trainIngredients)

#test/train split on training data
trainIngredients = trainIngredients[:4*trainIngredients.shape[0]//5]
testIngredients = trainIngredients[4*trainIngredients.shape[0]//5:]

trainCuisine = trainCuisine[:4*trainCuisine.shape[0]//5]
testCuisine = trainCuisine[4*trainCuisine.shape[0]//5:]

allIngredients = []
for i in range(len(trainIngredients)):
    for j in range(len(trainIngredients[i])):
        if trainIngredients[i][j] not in allIngredients:
            allIngredients.append(trainIngredients[i][j])

np.savez_compressed("raw_data", trainIngredients=trainIngredients, testIngredients = testIngredients, trainCuisine = trainCuisine, testCuisine = testCuisine, allIngredients = allIngredients, trainID = trainID, dtype = bool)
