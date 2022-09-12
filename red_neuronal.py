from tabnanny import verbose

import tensorflow as tf

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

def rnn(lista):
    # Datos de verdad

    #[graduado: 0|1, cod_ciudad: 000-1111, edad: 0, 2]

    td = []
    targetd = []
    for persona in lista:
        td.append([persona.edad, persona.cursosp])
        targetd.append([persona.retiro])

    training_data = np.array(td) # Estimulos               #   ENTRENAN RED

    target_data = np.array(targetd) # Acciones

    print(training_data)
    print(target_data)

    n_entrada = len(training_data[0]) #2

    n_nodos = 64

    n_salida = 1


    model = Sequential()

    model.add(Dense(n_nodos, input_dim = n_entrada, activation = "relu"))
    model.add(Dense(n_nodos, activation = 'sigmoid'))
    model.add(Dense(n_salida, activation = 'sigmoid'))



    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

    model.fit(training_data, target_data, epochs = 100, verbose = 1) # verbose = 0 es sin eco (no lo muestra en pantalla)

    real_data = np.array([[19,2]],'float32')

    resultado = model.predict(real_data, verbose=1)

    print(f'Entrada: {real_data[0]} => {resultado[0]}')
