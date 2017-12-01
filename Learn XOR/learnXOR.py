import numpy as np
from keras.models import *
from keras.layers import *

model = Sequential()
model.add(Dense(2, input_dim=2, activation='tanh'))         ## Modèle d'input
model.add(Dense(1, activation='sigmoid'))                   ## Modèle de sortie

model.compile(optimizer='adam', loss='binary_crossentropy') ## Modèle apprentissage

#model.load_weights('myXORModel.h5')

X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")          ## Tableau "Data" des entrées
Y = np.array([[0],[1],[1],[0]], "float32")                  ## Tableau "Pred" des résultats voulus

model.fit(X, Y, batch_size=1, epochs=10000, verbose=2)      ## Batch_size = "" ; epochs = "nombre d'epochs" ; verbose = "Manière de visualiser l'info (frequence affichage)"

model.save('myXORModel.h5')

print (model.predict(X))