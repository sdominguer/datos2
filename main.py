from datos import *
from metodosOrden import *
from generar import *
from red_neuronal import *

lista = generarDatos(1000)

'''
for elemento in lista:
    print(elemento)'''
    

print("Quick Sort")

lista = QuickSort(lista)

"""for elemento in lista:
    print(elemento)"""

rnn(lista)
