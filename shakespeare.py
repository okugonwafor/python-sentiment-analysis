#GENERATING POETIC TEXTS IN PYTHON

import random
import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop



filepath =tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
 
# TO SET OFF TEXT#

# we need to get the text of this file unto the script below

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

#we have to convert the sentences to numerical format because neural network is not able to deal with sentences

#we convert sentence to numerical to sentence

#select some part of the text

text = text[300000:800000]


#create a character set that contains all the possible characters that occur somewhere in the text (theoretically if character does not exist within the text does not appear in the set)

characters = sorted(set(text))

#sorted  

#create a dictionary that can connect characters as a key to numeric index as a value

char_to_index =dict((c, i) for i,c in enumerate(characters))

#eg {'a' =1, 'b' =5}

index_to_char = dict((i, c)for i,c in enumerate(characters))

#refer to number(key) and get the character
##
##
##

#PREDICT NEXT CHARACTER#

SEQ_LENGTH = 40 #use 40 characters to predict the next character

STEP_SIZE=  3 #how many characters are we going to shift before going to the next sequence, if seq_length = 5 and step_size =3 then we sample like this i.e "Hel-lo W-orl-d" - hello lo_Wo etc.

sentences = []
next_characters = []

#we want to feed characters to our neural network and predict the next word

for i in range (0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH]) 
    next_characters.append(text[i +SEQ_LENGTH]) #if the seq_length is 5 we get from 0:4 then index 5 is the next character

#fill up our lists - loops from beginning to end of our text 

#COVERT 'TRAINING DATA' INTO NUMPY OR TWO NUMPY ARRAYS
#note the values in the shapes of the array are coded in tuples (i.e parentheses) or you'll get too many arguments

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)

#we created numpy array with amount of sentences* legth of the sentences* amount of possible characters - with a data type boolean
# how it works one dimension for amount of sentences, one dimension for all the induvidual positions in these sentences and 1-d for all the characters can have
#whenever a specific character in a specific position occurs we set it to either true or 1 and the other values wii=ll remain 0

y = np.zeros((len(sentences), len(characters)), dtype = np.bool)

#we create two for loops. It is essentially to run the loop to locate a particular character in a sentence in a line of sentences and predict next character.

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

#build neural netwok trained to predict the next character
#LSTM - Long Short Term Memory will remember the past couple of characters
#Dense layer line of code is to add complexity to hidden layers by adding as many neurons as characters within the text
#softmax scales the output so that all values probabilities add up to 1 e.g 70% 'k' , 10% 'a' etc

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

#compile them all

model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.01))

# epochs = how many times our network is going to see the same data over and over again
model.fit(x,y, batch_size = 256, epochs =4)

model.save('textgenerator.model')

#function from keras tutorial. It takees the prediction and picks one character depending on temperature it uses softmax high temp-more creative, lo temp- less creative
#  
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        prediction = model.predict(x, verbose = 0)[0]
        #model is going to predict and we get softmax results of the prediction and feed to the sample function
        next_index = sample(prediction, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print('--------0.2--------')
print(generate_text(300, 0.2))
print('--------0.4--------')
print(generate_text(300, 0.4))
print('--------0.6--------')
print(generate_text(300, 0.6))
print('--------0.8--------')
print(generate_text(300, 0.8))
print('--------1--------')
print(generate_text(300, 1))