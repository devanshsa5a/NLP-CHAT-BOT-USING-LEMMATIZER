import tensorflow
import numpy 
import tflearn

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer()


from gtts import gTTS 
  
import os 
language = 'en'

with open("intents.json") as file:
    data = json.load(file)
try:
 
    with open("data.pickle", "rb") as f:
       words, labels, training, output = pickle.load(f)
except:
      words = []
      labels = []
      docs_x = []
      docs_y = []

      for intent in data["intents"]:
              for pattern in intent["patterns"]:
                  wrds = nltk.word_tokenize(pattern)
                  words.extend(wrds)
                  docs_x.append(wrds)
                  docs_y.append(intent["tag"])

              if intent["tag"] not in labels:
                  labels.append(intent["tag"])

      words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
      words = sorted(list(set(words)))

      labels = sorted(labels)

      training = []
      output = []

      out_empty = [0 for _ in range(len(labels))] 

      for x, doc in enumerate(docs_x):
              bag = []

              wrds = [lemmatizer.lemmatize(w.lower()) for w in doc if w not in "?"]

              for w in words:
                  if w in wrds:
                      bag.append(1)
                  else:
                      bag.append(0)

              output_row = out_empty[:]
              output_row[labels.index(docs_y[x])] = 1

              training.append(bag)
              output.append(output_row)


      training = numpy.array(training)
      output = numpy.array(output)

      with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 25)
net = tflearn.fully_connected(net, 25)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net) 

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def binary_of_data(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chatbot():
    print("Start talking with the bdcoe bot (type quit to stop)!")
    while True:
        print(" ")
        inp = input("bdcoe bot: ")
        if inp.lower() == "quit":
          print(" ")
          print("Have a good day.")
          print("__"*40)
          print(" ")
          final="espeak 'Have a good day. '"
          os.system(final) 
          break

        results = model.predict([binary_of_data(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.5 :

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            text=random.choice(responses)
            print(" ")
            print(text)
            print("__"*40)
            print(" ")
            final="espeak '{}'"

            os.system(final.format(text)) 

        else:

            print(" ")
            print("I Didn't Understand your question.Please try different question or check your spellings.")
            print("__"*40)
            print(" ")    

            final="espeak 'I Didnot Understand your question.Please try different question or check your spellings.'"

            os.system(final) 
               

 
chatbot()