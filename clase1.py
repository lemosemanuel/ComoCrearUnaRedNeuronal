import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.engine import sequential


# 7 pasos simples para crear una Red Neuronal Por Emanuel LEmos 
# Paso 1: Inicializar los pesos aleatoriamente con valores cercanos a 0 (pero no 0).
# Paso 2: Introducir la primera observación del dataset en la Capa de Entrada, cada característica es un nodo de entrada.
# Paso 3: Propagación hacia adelante: de izquierda a derecha, las neuronas se activan de modo que la activación de
# cada una se limita por los pesos. Propaga las activaciones hasta obtener la predicción y.
# Paso 4: Comparamos la predicción con el resultado real. Se mide entonces el error generado.
# Paso 5: Propagación hacia atrás: de derecha a izquierda, propagando el error hacia atrás. Se actualizan los pesos
# según lo responsables que Sean del error. El ratio de aprendizaje gobierna cuánto deben actualizarse los pesos.
# Paso 6: Se repiten los Pasos 1 a 5 y se actualizan los pesos después de cada observación (Reinforcement Learning). O:
# Se repiten los Pasos 1 a 5 pero actualiza los pesos después de un conjunto de observaciones (Batch Learning).
# Paso 7: Cuando todo el Conjunto de Entrenamiento ha pasado por la RNA, se completa un epoch. Hacer más epochs.



df = pd.read_csv('Churn_Modelling.csv')
# me quedo solo con lo que me sirve del df
#me quedo desde la columna 3 hasta la 12
X = df.iloc[:, 3:13].values
Y = df.iloc[:, 13].values

#tengo variables categoricas , las modifico
#las paso en dummies (con OneHotEncoder y LabelEncoder)

# el LabelEconder lo que hace es modifica la variable en numerico , ej: si hay 3 variables lo modifica en :[1,2,3]
#creo el label encoder para el pais
'''labelencoder_X_1 = LabelEncoder()
X.iloc[:,1]=labelencoder_X_1.fit_transform(X.iloc[:,1])
#ahora con el genero
labelenconder_X_2=LabelEncoder()
X.iloc[:,2]=labelenconder_X_2.fit_transform(X.iloc[:,2])
'''


# podria hacerlo mas facil con una funcion que me busque las columnas que son string y me las convierta

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# hago el train test 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#hago el escalado de variables (ya que hay mucha diferencia entre variables)
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)



#CONSTRUIR LA RNA
# recordar los 7 pasos
#importo librerias Keras y adicionales
import keras
# inicializar los parametros con Sequential
from keras.models import Sequential
# Dense para crear las capas intermedias de la red neuronal
from keras.layers import Dense

# inicializar la red neuronal artificial
# como estamos en un problema de clasificacion vamos a usar classifier
classifier = Sequential()
# capas de entrada y primera capa oculta de nuestra red neuronal

# agrego la capa (lo hago con Dense)
# units es el numero de nodos que vamos a agregar a la capa oculta
# pero cuantos nodos pongo ? normalmente (y no siempre es asi) se suele utilizar la media entre las entradas y la salida , si en entrada tengo 11 y salida 1 , entonces seran 6
#kernel_initializer=como vamos a inicializar esos pesos ?
#activation= funcion de activacion (sigmoide,,relu(rectificador lineal unitario) etc)
#input_dim= cantidad de nodos de entrada
# osea que la capa de aca tiene 6 unidades de salida y 11 de entrada
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 11))
# ya tenemos la primera capa oculta
# vamos a crear otra
# saco el input dim en ya que la red neuronal ya sabe cuanto va a entrar
# esta capa tiene 6 de entrada y 6 de salida
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))


# creo la capa de salida
# esta tiene 6 de entrada y obviamente (al ser una respuesta sola, ojo en otras redes puede haber mas de una) 1 solo nodo en la de salida
# cambio la funcion de activacion (y como me interesa que sea una 
# probabilidad entonces voy a usar una Sigmoide)
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))



# ahora hay que compilar toda la red neuronal
# metodo de compilacion "compile"
# optimazer es el metodo de otimizacion que vamos a utilizar
# loss= funcion de perdida (como vamos a medir el error)
# metrics= medidas de precision
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])



# vamos a la parte del entrenamiento
# epochs numero de iteraciones globales (esta bueno que sean varias pero no muchos CUIDADO OVERFITING)
# batch_size procesar varios bloques y luego corregir
classifier.fit(X_train, Y_train,  batch_size = 10, epochs = 100)

#predecir a los usuarios que no conoce ,
#  se van o no se van del banco?
y_pred=classifier.predict(X_test)

y_pred


#hago una matris de confucion
from sklearn.metrics import confusion_matrix
# como habia hecho una sigmoide tengo que marcar un umbral que me dice si el cliente abandona o no
y_pred=(y_pred>0.5)
cm= confusion_matrix(Y_test,y_pred)

cm

(1515+204)/2000


### PERO OJO HAY QUE INTENTAR RESOLVER EL PROBLEMA DE LA VARIANZA ,
#  ya que sino cada vez que lo entrene me va a dar diferente
# ver clase 2
