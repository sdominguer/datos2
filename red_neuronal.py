from tabnanny import verbose
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from generar import *

# Datos de verdad
#[graduado: 0|1, cod_ciudad: 000-1111, edad: 0, 2]
training_data = np.array([[contador_lista_anos, ultimo_ano, indice],'float32']) # Estimulos               #   ENTRENAN RED
target_data = np.array([[estado_var], 'float32']) # Acciones

n_entrada = len(training_data[0]) #2
n_nodos = 16
n_salida = 1

model = Sequential()
model.add(Dense(n_nodos, input_dim = n_entrada, activation = "relu"))
model.add(Dense(n_salida, activation = 'sigmoid'))
model.compile(
    loss = "mean_squared_error",
    optimizer = "adam", 
    metrics = ['binary_accuracy']
)
model.fit(training_data, target_data, epochs = 1000, verbose = 0) # verbose = 0 es sin eco (no lo muestra en pantalla)


#==============================================================================

print('='*30)
score = model.evaluate(training_data, target_data, verbose = 1)
print (f'Modelo de medicion: {model.metrics_names[1]}')
print(f' ... valor del modelo: {score[1] * 100}')
model.summary()

#==============================================================================

real_data = np.array([[contador_lista_anos, ultimo_ano, indice],"float32"])
resultado = model.predict(real_data, verbose = 1)
print(f'Entrada: {real_data[0]} => {resultado[0]}')
#0.73724
