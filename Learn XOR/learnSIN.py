import numpy as np
from keras.models import *
from keras.layers import *

model = Sequential()
model.add(Dense(2, input_dim=shape, activation='tanh'))         ## Modèle d'input
model.add(Dense(1, activation='sigmoid'))                   ## Modèle de sortie

model.compile(optimizer='adam', loss='binary_crossentropy') ## Modèle apprentissage

model.load_weights('mySinModel.h5')

X = np.arange(100)          ## Tableau "Data" des entrées
Y = np.sin(X)                  ## Tableau "Pred" des résultats voulus

for i in range(10):
    model.fit(X, Y, batch_size=1, epochs=10, verbose=2)      ## Batch_size = "" ; epochs = "nombre d'epochs" ; verbose = "Manière de visualiser l'info (frequence affichage)"
#    model.save('myXORModel%d.h5' % i)                       ## Saves before doing another loop

model.save('mySinModel.h5')

print (model.predict(X))