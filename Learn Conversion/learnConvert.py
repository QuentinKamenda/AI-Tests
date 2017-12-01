import numpy as np
from keras.models import *
from keras.layers import *

model = Sequential([Dense(1, input_shape=(3,), activation='relu')])

model.compile(optimizer='SGD',metrics=['mean_squared_error'], loss='mean_squared_error') ## Modèle apprentissage

#model.load_weights('myConvertModel.h5')

X = np.array([[0,0,1],[0,1,1],[1,1,0]], 'float32')      ## Tableau "Data" des entrées
Y = np.array([[1],[3],[6]], "float32")                  ## Tableau "Pred" des résultats voulus

for i in range(1):
    model.fit(X, Y, batch_size=3, epochs=1000, verbose=1)      ## Batch_size = "nombre de données qu'il traite par epoch" ; epochs = "nombre d'epochs" ; verbose = "Manière de visualiser l'info (frequence affichage)"

model.save('myConvertModel.h5')

print (np.round(model.predict(X)))

deux = np.array([[0,1,0]])
print (np.round(model.predict(deux)))

for layer in model.layers:
    weights = layer.get_weights()
    print (np.round(weights[0]))