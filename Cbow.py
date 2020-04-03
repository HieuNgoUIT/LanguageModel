import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import re
def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())
def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()
    for i,word in enumerate(set(tokens)):
        word_to_id[word] = i
        id_to_word[i] = word
    return word_to_id, id_to_word

doc = """
Mới đây, VFF và VPF dự kiến tổ chức cuộc họp vào ngày 31/3 với đại diện các CLB để thống nhất phương án tổ chức V.League ngay sau khi dịch bệnh lắng xuống. Họ đảm bảo chỉ tổ chức giải khi Chính phủ cho phép, trong điều kiện an toàn và trước mắt vẫn thi đấu ở trên sân không khán giả cho đến khi dịch bệnh hết hẳn. 

Một động thái được cho là bình thường nhưng bầu Đức lại thể hiện sự tức giận. Ông tiếp tục đưa ra tuyên bố "cấm HAGL tham gia bất kỳ hoạt động thể thao nào trong giai đoạn này", thậm chí, không tham gia họp kể cả trực tiếp hay trực tuyến của VFF và VPF
"""
tokenszero = [""]
tokens = tokenize(doc)
tokens = tokenszero + tokens
vocab = set(tokens)
print(len(tokens))
print(len(vocab))
print(tokens)

word_to_id, id_to_word = mapping(tokens)
#print(word_to_id)
#print(id_to_word)

def generate_training_data(sentence, word_to_id, window_size):
    L = len(sentence)
    X, Y = [], []
    tempX, tempY = [], []
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
            tempX.insert(0,0)
            filling_missing_left +=1
        while(filling_missing_right < 3):
            tempX.append(0)
            filling_missing_right +=1
        X.append(tempX)
        Y.append(word_to_id[sentence[i]])
        tempX = []
    return X,Y

X,Y = generate_training_data(tokens, word_to_id, 3)
print(X[:10])
print(type(X))
print("------------------------------------------------------")
print(Y[:10])
#print(Y.shape)

MAX_LENGTH = 6

X = pad_sequences(X, maxlen=MAX_LENGTH, padding='post')
print(X.shape)

vocab_size = len(vocab)
Y = to_categorical(Y, num_classes=vocab_size)
print(Y.shape)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=6))
model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
print(model.summary())



model.fit(X, Y, epochs=100, verbose=2)

def spell_checking(sentence):
    X, _ = generate_training_data(sentence, word_to_id, 3)
    for i in range(len(X)):
        yhat = model.predict_classes(np.array([X[i]]))
        word = id_to_word[int(yhat)]
        print(i)
        print(word)
        print("_________")
        
X, _ =spell_checking(['dự', 'kiến', 'tổ', 'chức', 'họp','vào'])