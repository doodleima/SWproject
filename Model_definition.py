# 모델 및 데이터셋 생성 정의 메소드
import numpy as np

from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dropout, GlobalMaxPool1D

# 데이터셋 생성
def dataset(data, label, size):
    feature_list = []
    label_list = []
    for i in range(len(data) - size):
        feature_list.append(np.array(data.iloc[i:i+size]))
        label_list.append(np.array(label.iloc[i+size]))

    return np.array(feature_list), np.array(label_list)

# NLP 데이터 모델
def NLP_Model(total_words_len) :
    model = Sequential()
    model.add(Embedding(total_words_len+1, 128, input_length = 30))
    model.add(Dropout(0.1))
    model.add(Conv1D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu')) # 768, 3
    model.add(GlobalMaxPool1D())
    model.add(Dense(64)) # , activation='relu'
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    return model

# 주가 모델
def Stock_Model(feature) :
    model = Sequential()
    model.add(LSTM(16, input_shape = (feature.shape[1], feature.shape[2]), activation='relu', return_sequences=False))
    model.add(Dense(1)) # , activation='sigmoid')

    return model
