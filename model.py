import numpy as np
import json

#TRAINING SET
trainData = json.load(open('train.json'))

trainCuisine = []
trainID = []
trainIngredients = []
for i in range(len(trainData)):
    cuisine = trainData[i]["cuisine"]
    trainCuisine.append(cuisine)
    identification = trainData[i]["id"]
    trainID.append(identification)
    ingredients = trainData[i]["ingredients"]
    #TODO: clean special characters in ingredients
    trainIngredients.append(ingredients)


trainCuisine = np.array(trainCuisine)
trainID = np.array(trainID)
trainIngredients = np.array(trainIngredients)

#TEST SET
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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

#print("trainIngredients.shape[1],", trainIngredients, (trainIngredients.shape[0],))
#TODO: build neural net, use dense layers, use ensemble learning method: bagging
#countvectorizer
'''
vectorizer = CountVectorizer(input = "content")
boolIngredients = []
for i in range(len(trainIngredients)):
    vectorIngredients = vectorizer.fit(trainIngredients[i])
    boolIngredients.append(vectorIngredients)

boolIngredients = np.array(boolIngredients)
print("vectorrr", boolIngredients)
'''
vectorizer = CountVectorizer(input = "content")
boolIngredients = []
for i in range(len(trainIngredients)):
    vectorIngredients = vectorizer.fit_transform(trainIngredients[i])
    boolIngredients.append(vectorIngredients)

boolIngredients = np.array(boolIngredients)

print("vectorrr", boolIngredients.shape)

neural_net = Sequential()
neural_net.add(Dense(200, activation='relu', input_shape = (boolIngredients.shape[0],)))
neural_net.add(Dropout(0.1))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(50, activation='softmax'))
neural_net.summary()

neural_net.compile(optimizer="Adamax", loss="categorical_crossentropy", metrics=['accuracy'])
history = neural_net.fit(boolIngredients, trainCuisine, verbose=1, epochs=10)



"""#TODO: build a model"
neural_net = Sequential()
neural_net.add(Conv2D(5, (2, 2), activation='relu'))
neural_net.add(Dropout(0.1))
neural_net.add(Conv2D(5, (2, 2), activation='relu'))
neural_net.add(Flatten())
neural_net.add(Dense(64, activation='softmax'))

neural_net.summary()
#TODO: train the model"

neural_net.compile(optimizer="Adamax", loss="categorical_crossentropy", metrics=['accuracy'])
history = neural_net.fit(train_X, train_Y, verbose=1, validation_data=(test_X, test_Y), epochs=10)

loss, accuracy = neural_net.evaluate(test_X, test_Y, verbose=0)
print("accuracy: {}%".format(accuracy*100))

#TODO: test the model

for i in range(10):
    index = np.random.randint(test_X.shape[0])
    shiftIndex = index+(4*states.shape[0] // 5)

    #correct answer
    print(show_move(test_X[index], moves[shiftIndex], turns[shiftIndex]))
    print(show_values(test_X[index], values[shiftIndex]))
    print("shiftIndex", moves[shiftIndex])
    predictedMove = neural_net.predict(test_X)[index]
    moveIndex = np.argmax(predictedMove)
    movePair = (moveIndex//8, moveIndex-(moveIndex//8)*8)
    print("movepair", movePair)
    #prediction
    print(show_move(test_X[index], movePair, turns[shiftIndex]))
    print(show_values(test_X[index], values[shiftIndex]))

"""
