import numpy as np
import json

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

"""
testData = json.load(open('test.json'))

testID = []
testIngredients = []
for i in range(len(testData)):
    identification = testData[i]["id"]
    testID.append(identification)
    ingredients = testData[i]["ingredients"]
    testIngredients.append(ingredients)

testID = np.array(testID)
testIngredients = np.array(testIngredients)
"""
from sklearn.feature_extraction.text import CountVectorizer

#### TRAIN ####
allIngredients = []
for i in range(len(trainIngredients)):
    for j in range(len(trainIngredients[i])):
        if trainIngredients[i][j] not in allIngredients:
            allIngredients.append(trainIngredients[i][j])

boolTrainIngredients = np.zeros((len(trainIngredients), len(allIngredients)))
for i in range(len(trainIngredients)):
    recipeIngredients = trainIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTrainIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolTrainCuisine = vectorizer.fit_transform(trainCuisine).toarray()
boolTrainCuisine = boolTrainCuisine.argmax(1)

#### TEST ####

#use original testIngredients
boolTestIngredients = np.zeros((len(testIngredients), len(allIngredients)))
for i in range(len(testIngredients)):
    recipeIngredients = testIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTestIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolTestCuisine = vectorizer.fit_transform(testCuisine).toarray()
boolTestCuisine = boolTestCuisine.argmax(1)

np.savez_compressed("data", trainX=boolTrainIngredients, trainY = boolTrainCuisine, testX = boolTestIngredients, testY = boolTestCuisine)
