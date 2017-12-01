# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

import numpy as np
from keras.models import *
from keras.layers import *

dataset = np.loadtxt("pima-indians-diabetes.data", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))         ## Modèle 1ere row (avec input)
model.add(Dense(8, activation='relu'))                      ## Modèle de la 2nde row
model.add(Dense(1, activation='sigmoid'))                   ## Modèle de sortie

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       ## On compile le modèle

model.fit(X, Y, epochs=10000, batch_size=200, verbose=0.1)       ## On le fait apprendre



