# Main chatbot script

## Imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle 

## Import the json of intents
with open('intents.json') as file:
    data = json.load(file)

## Preprocessing (edited to use pickle for speed!)
try:
    # Loading pickle
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # Alllll the preprocessing steps
    words = [] # list of unique words
    docs_x = [] # word sequences
    docs_y = [] # tag for modeling purposes
    labels = [] # tag

    for intent in data['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Stemming out the words list
    stemmer = LancasterStemmer()
    words = [stemmer.stem(w.lower()) for w in words if w not in '?'] # removes question mark
    words = sorted(list(set(words)))  # removes duplicates

    # Sorting labels for fun
    labels = sorted(labels)

    # Create bag of words for training model
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]  # Creates a list of 0's

    # Actually creating bag of words
    for x, doc in enumerate(docs_x):
        bag = []
        word_list = [stemmer.stem(w) for w in doc]
        # Populate the list
        for w in words:
            if w in word_list:
                bag.append(1)
            else:
                bag.append(0)
        # Generating output
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        # Filling training & output
        training.append(bag)
        output.append(output)

    # Changing training & output to numpy arrays
    training = np.array(training)
    output = np.array(output)

    # Saving data with pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)


## Modeling

# Setting up neural network with tflearn
net = tflearn.input_data(shape = [None, len(training[0])])  # input layer
net = tflearn.fully_connected(net, 8)  # hidden layer
net = tflearn.fully_connected(net, 8) # another hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax') # output layer
net = tflearn.regression(net)  # adds regression
# Compile 
model = tflearn.DNN(net)

# Load saved model or fit new model
try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save('model.tflearn')


## Let's get this working!

# Function to create bag of words based on user input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  # another list of 0's
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sentence in s_words:
        for i, w in enumerate(words):
            if w == sentence:  # if current word equals a word in a sentence
                bag[i] = 1

    return np.array(bag)  # returns an array of the bag of words


# Function for actually chatting with the model
def chat():
    print("Start talking to the bot!  Type 'Quit' to stop.")

    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        # Using the model on user input
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)  # returns most probable response
        tag = labels[results_index]  # returns tag for most probable

        # Use tag from json to return a random response, should one exist
        if results[results_index] > 0.7:  # better than 70% probability
            for tg in data['intents']:
                if tg[tag] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't understand that.  Please try again or ask something else.")


# Run the chatbot
chat()