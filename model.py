import numpy as np
import json
#from view_hex_data import show_board, show_move, show_values


data = json.dumps("train.json")
#data = np.load("train.json")

with open("train.json") as data:
    id = data["id"]
    cuisine = data["cuisine"]
    ingredients = data["ingredients"]

print("ingredients:", ingredients, ingredients.shape)


'''

index = np.random.randint(states.shape[0])
print(show_move(states[index], moves[index], turns[index]))
print(show_values(states[index], values[index]))

#Split into training and test sets
train_X = states[:4*states.shape[0] // 5]
test_X = states[4*states.shape[0] // 5:]

print("train_X:", train_X.shape)
print("test_X:", test_X.shape)

data = [] # 6614 by 64
for i in range(moves.shape[0]):
    array = np.zeros(64) # 1 by 64
    oneMove = moves[i]
    index = oneMove[0]*8 + oneMove[1]
    array[index] = 1
    data.append(array)

data = np.array(data)

#TODO: Create train_Y and test_Y. What do we want the targets to be?
train_Y = data[:4*states.shape[0] // 5]
test_Y = data[4*states.shape[0] // 5:]

print("train_Y:", train_Y.shape)
print("test_Y:", test_Y.shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout

#TODO: build a model"
neural_net = Sequential()
neural_net.add(Conv2D(5, (2, 2), activation='relu', input_shape=(8, 8, 1)))
neural_net.add(Dropout(0.1))
neural_net.add(Conv2D(5, (2, 2), activation='relu', input_shape=(8, 8, 1)))
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

'''
