from datos import *
from metodosOrden import *
import matplotlib.pyplot as plt

lista = generarDatos(10)

'''
for elemento in lista:
    print(elemento)'''
    

print("Quick Sort")

lista = QuickSort(lista)

for elemento in lista:
    print(elemento)
