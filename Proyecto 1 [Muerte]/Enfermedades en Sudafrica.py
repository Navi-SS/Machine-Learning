#|===============================================================|
#|Proyecto #1 Calcular enfermedades cardiovasculares en Sudafrica|
#|===============================================================|

#----------------------
#Importar las librerias
#----------------------

import pandas as pd
import matplotlib.pyplot as plt

#Importar los datos
data=pd.read_csv('phpgNaXZe.csv')
print(data.head())

#Colocar los nombres a las columnas
Columnas=["Sbp","Tabaco","Ldl","Adiposity","Familia","Tipo","Obesidad","Alcohol","Edad","Chd"]
data.columns=Columnas #Se modifica el nombre de las columnas
print(data.head())

#-------------------
#Verificar los datos
#-------------------

#Conocer el tipo de datos de la data
print(data.dtypes)
#La mayoria de los datos son entero y flotante, por lo cual no se requiere algun cambio

#Verificar si no hay datos perdidos
print(data.isnull().sum())
#Se notificará la suma total de datos nulos, no hay datos perdidos

#----------------------
#Procesamiento de datos
#----------------------

#Los datos de familia y chd son valores 1 y 2, se cambiaran por 0 y 1
#Cambiar los datos de las columnas familia y chd
from sklearn.preprocessing import LabelEncoder #Modifica los resultados de una etiqueta

encoder=LabelEncoder()
data["Familia"]=encoder.fit_transform(data["Familia"])
data["Chd"]=encoder.fit_transform(data["Chd"])
print(data.head())
#Donde antes habia un 1 se cambio por 0 y el 2 cambio a 1

#Escalamos los valores de la columna Sbp
from sklearn.preprocessing import MinMaxScaler
#Con esta función se establecen un minimo y maximo de valores definidos y los transforma a este rango
scale=MinMaxScaler(feature_range=(0,100))
data["Sbp"]=scale.fit_transform(data["Sbp"].values.reshape(-1,1))
print(data.head())

#----------------------
#Visualización de datos
#----------------------

#Visualizar la obesidad de acuerdo a la edad
data.plot(x="Edad",y="Obesidad",title="Edad vs Obesidad",kind="scatter",figsize=(10,5))
#Visualizar el tabaco consumido de acuerdo a la edad
data.plot(x="Edad",y="Tabaco",title="Edad vs Tabaco consumido",kind="scatter",figsize=(10,5))
#Visualizar el alcohol consumido de acuerdo a la edad
data.plot(x="Edad",y="Alcohol",title="Edad vs Alcohol consumido",kind="scatter",figsize=(10,5))
plt.show()

#----------------------------
#Análisis de Machine Learning
#----------------------------

#Separar datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
#Importamos el modelo de maquinas de vectores de soporte, por que es un problema de clasificación
from sklearn import svm
#Importamos mas funciones para trabajar
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

#Definir variable dependiente e independiente
y=data["Chd"] #Y sera la columna Chd
X=data.drop("Chd",axis=1) #X el resto de valores del conjunto de datos

#Separar los datos de entrenamiento y prueba
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Definir el algoritmo a utilizar
algoritmo=svm.SVC(kernel="linear")

#Definir el algoritmo
algoritmo.fit(X_train,y_train)

#Realizar la predicción
y_test_pred=algoritmo.predict(X_test)

#Se calcula la matriz de confusión
print(confusion_matrix(y_test,y_test_pred))
#Los datos correctos son [a11]+[a22], diaognal principal
#Los datos incorrectos son [a12]+[a21], diaonal secundaria

#Se calcula la exactitud del modelo
print(accuracy_score(y_test,y_test_pred))
#Se calcula la presición del modelo
print(precision_score(y_test,y_test_pred))