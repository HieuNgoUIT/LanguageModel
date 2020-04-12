import os
from sys import getsizeof
import re
import json
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import  Dense, GRU, Embedding
from pyvi import ViTokenizer
import string
from collections import Counter 



root = "./mysmalltest/"
print("Loading data from", root)
entries = os.listdir(root)
data = ""
for entry in entries:
    path = root + entry
    try:
        with open(path) as f:
            currentdata = f.read()
        data = data +  currentdata
    except:
        print("This", path, "is not working") 
print("Done loading data of", getsizeof(data), "bytes")


def replace(chuoi):
    return chuoi.replace("_", " ")
data = data.translate(str.maketrans('', '', string.punctuation))
tokens = ViTokenizer.tokenize(data).lower().split()
tokens = list(map(replace, tokens))
print("length tokens",len(tokens))

  
def removeElements(lst, k): 
    counted = Counter(lst) 
    return [el for el in lst if counted[el] > k] 
      
training_tokens = removeElements(tokens, 3)
print("length training tokens",len(training_tokens))


def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()
    for i,word in enumerate(set(tokens)):
        word_to_id[word] = i
        id_to_word[i] = word
    return word_to_id, id_to_word
tokens_zero = "<empty>"
if tokens_zero not in training_tokens:
    training_tokens.insert(0,tokens_zero)
vocab = set(training_tokens)
vocab_size = len(vocab)
print("token length",len(training_tokens))
print("vocab length",vocab_size)
print("Mapping to create vocabulary")
word_to_id, id_to_word = mapping(training_tokens)


print("Export word_to_id and id_to_word file")
with open('word_to_id.json', 'w') as f:
    json.dump(word_to_id, f, ensure_ascii=False)
with open('id_to_word.json', 'w') as f:
    json.dump(id_to_word, f, ensure_ascii=False)


def generate_training_data(sentence, word_to_id, window_size):
    L = len(sentence)
    X, Y = [], []
    tempX= []
    for i in range(L):
        index_before_target = list(range(max(0, i - window_size), i))           
        index_after_target = list(range(i + 1, min(i + window_size + 1,L)))           
        index_before_after_target = index_before_target + index_after_target
        #print(index_before_after_target)                     
        for j in index_before_after_target:
            tempX.append(word_to_id[sentence[j]])
        #print(tempX)
        filling_missing_left = len(index_before_target)
        filling_missing_right = len(index_after_target)
        while(filling_missing_left < 3):
            tempX.insert(0, word_to_id['<empty>'])
            filling_missing_left +=1
        while(filling_missing_right < 3):
            tempX.append(word_to_id['<empty>'])
            filling_missing_right +=1
        X.append(tempX)
        Y.append(word_to_id[sentence[i]])
        tempX = []
    return np.array(X),Y

print("Generating training data from corpus")
X,Y = generate_training_data(training_tokens, word_to_id, 3)
print("length X:",len(X))
print("length Y:",len(Y))
print(type(X))
print("X shape",X.shape)
print(X[:5,:])

Y = to_categorical(Y, num_classes=vocab_size)
print("Y shape",Y.shape)


def handle_key_range(values):
    l = len(values)
    char_key = l - 100
    word = ''
    for i in range(char_key):
        word += " "+ values[i]
    coefs = np.asarray(values[char_key:], dtype='float32')
    return word.strip(), coefs

print("Loading word2vec")
embeddings_index = dict()
count = 0
with open('/media/hieu/CA48F77D48F7669B/SpellChecking/model.txt',"r", encoding='utf-8') as f:
    f.readline() #skip first row  
    while True:
        try:
            line = next(f)
            values = line.split()
            word, coefs = handle_key_range(values)
            embeddings_index[word] = coefs
            count += 1
            if count == 10:
                break
        except:
            print("Line is broken")
print('Loaded %s word vectors.' % len(embeddings_index))

print("Create a weight matrix for words in trainning docs")
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in word_to_id.items():
    #print(word, i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        try:
            embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = np.zeros(100)


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=6, weights=[embedding_matrix], trainable=False))
model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
print(model.summary())


model.fit(X, Y, epochs=1, verbose=2)
model.save("testing.h5")
print("Saved model to disk")


def encode_input(input_string):
    return input_string.split(" ")


def spell_checking(sentence):
    encode_sentence = encode_input(sentence)
    X, y = generate_training_data(encode_sentence, word_to_id, 3)
    error_words = []
    for i in range(len(X)):
        word = encode_sentence[i]
        word_index = word_to_id[word]
        yhat = model.predict_proba(np.array([X[i]]))
        if yhat[:,word_index] < 0.1:
            error_words.append(encode_sentence[i])
        print("input:",np.array([X[i]]))
        print("index:",i)
        print("probability of ---- ", encode_sentence[i], "----given surrouding is:" ,yhat[:,word_index])
        print("__________________________________________________________________________")
    return error_words

print("empty index:",word_to_id['<empty>'])
spell_checking("chiến tranh thế giới")
spell_checking("chiến tránh thế giới")



