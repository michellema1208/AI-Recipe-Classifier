import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


#### TRAIN ####
loaded = np.load("raw_data.npz")
trainIngredients = loaded["trainIngredients"]
trainCuisine = loaded["trainCuisine"]
allIngredients = loaded["allIngredients"]

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
test = np.load("test.npz")
boolTestIngredients = test["boolTestIngredients"]
boolTestCuisine = test["boolTestCuisine"]


#### NEURALNET ####
neural_net = Sequential()
neural_net.add(Dense(100, activation='relu', input_shape = (boolTrainIngredients.shape[1],)))
neural_net.add(Dropout(0.1))
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(20, activation='softmax'))
neural_net.summary()

neural_net.compile(optimizer="Adamax", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = neural_net.fit(boolTrainIngredients, boolTrainCuisine, verbose=1, validation_data=(boolTestIngredients, boolTestCuisine), epochs=10)
yTest = neural_net.predict(boolTestIngredients)
predictLabels = np.argmax(yTest, axis = 1)
print("#### NEURALNET ####")
for i in range(len(predictLabels)):
    for j in range(len(boolTrainCuisine)):
        if boolTrainCuisine[j] == predictLabels[i]:
            cuisineLabel = trainCuisine[j]
    print(i, "predicted label:", cuisineLabel)
loss, accuracy = neural_net.evaluate(boolTestIngredients, boolTestCuisine, verbose=0)
print("NN accuracy: {}%".format(accuracy*100))

#### RANDOMFOREST ####
forest = RandomForestClassifier(n_estimators=500, max_features='auto', class_weight='balanced')
forest.fit(boolTrainIngredients, boolTrainCuisine)
yTest = forest.predict(boolTestIngredients)
#predictLabels = np.argmax(yTest, axis = 1)
print("#### RANDOMFOREST ####")
for i in range(len(predictLabels)):
    for j in range(len(boolTrainCuisine)):
        if boolTrainCuisine[j] == predictLabels[i]:
            cuisineLabel = trainCuisine[j]
    print(i, "predicted label", cuisineLabel)
accuracy = forest.score(boolTestIngredients, boolTestCuisine)
print("RF accuracy: {}%".format(accuracy*100))


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
