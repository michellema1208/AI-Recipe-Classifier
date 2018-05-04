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

'''
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
print(clf.feature_importances_)
[ 0.17287856  0.80608704  0.01884792  0.00218648]
print(clf.predict([[0, 0, 0, 0]]))
'''
'''
X, y = make_classification(n_samples=boolIngredients.shape[0], n_features=4, n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)

'''

#build neural net, use dense layers
#### TRAIN ####
allIngredients = []
for i in range(len(trainIngredients)):
    for j in range(len(trainIngredients[i])):
        if trainIngredients[i][j] not in allIngredients:
            allIngredients.append(trainIngredients[i][j])

boolIngredients = np.zeros((len(trainIngredients), len(allIngredients)))
for i in range(len(trainIngredients)):
    recipeIngredients = trainIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolIngredients[i][j] = 1

vectorizer = CountVectorizer(input = "content")
boolCuisine = vectorizer.fit_transform(trainCuisine).toarray()
boolCuisine = boolCuisine.argmax(1)

#### TEST ####
'''
allTestIngredients = []
for i in range(len(testIngredients)):
    for j in range(len(testIngredients[i])):
        if testIngredients[i][j] not in allTestIngredients:
            allTestIngredients.append(testIngredients[i][j])
'''
#use original testIngredients
boolTestIngredients = np.zeros((len(testIngredients), len(allIngredients)))
for i in range(len(testIngredients)):
    recipeIngredients = testIngredients[i]
    for j in range(len(allIngredients)):
        if allIngredients[j] in recipeIngredients:
            boolTestIngredients[i][j] = 1

neural_net = Sequential()
neural_net.add(Dense(100, activation='relu', input_shape = (boolIngredients.shape[1],)))
neural_net.add(Dropout(0.1))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(20, activation='softmax'))
neural_net.summary()

neural_net.compile(optimizer="Adamax", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = neural_net.fit(boolIngredients, boolCuisine, verbose=1, epochs=10)
yTest = neural_net.predict(boolTestIngredients)
predictLabels = np.argmax(yTest, axis = 1)
print("NEURALNET##############")
print(predictLabels)
for i in range(len(predictLabels)):
    for j in range(len(boolCuisine)):
        if boolCuisine[j] == predictLabels[i]:
            cuisineLabel = trainCuisine[j]
    print(testID[i], "predicted label", cuisineLabel)


#build random forest
print("43570")
forest = RandomForestClassifier(n_estimators=500, max_features='auto', class_weight='balanced')
print("hi")
forest.fit(boolIngredients, boolCuisine)
print("hi")
yTest = forest.predict(boolTestIngredients)
#predictLabels = np.argmax(yTest, axis = 1)
print("RANDOMFOREST##############")
print(yTest)
for i in range(len(predictLabels)):
    for j in range(len(boolCuisine)):
        if boolCuisine[j] == predictLabels[i]:
            cuisineLabel = trainCuisine[j]
    print(testID[i], "predicted label", predictLabels[i])

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
