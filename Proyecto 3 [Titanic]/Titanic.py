#Predecir la supervivencia del Titanic

#|===================================|
#|Importamos las librerias a utilizar|
#|===================================|

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#Librerias de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#|==================|
#|Preparamos la Data|
#|==================|

train=pd.read_csv('train.csv')
print(train.head())
test=pd.read_csv('test.csv')
print(test.head())

#Verificar la cantidad de datos en el dataset
print("\nCantidad de datos: ")
print(train.shape)
print(test.shape)

#Verificamos la información que hay en el Dataset
print("\nTipos de datos: ")
print(train.info())
print(test.info())

#Verificamos si hay algún dato faltante
print("\nDatos faltantes: ")
print(pd.isnull(train).sum())
print(pd.isnull(test).sum())

#Descripción del data set
print("\nEstadisticas del Dataset: ")
print(train.describe())
print(test.describe())

#Cambiar el sexo de los pasajeros
train['Sex'].replace(['female','male'],[0,1],inplace=True)
test['Sex'].replace(['female','male'],[0,1],inplace=True)

#Cambiamos los datos de embarque por numeros
train['Embarked'].replace(['Q','S','C'],['0','1','2'],inplace=True)
test['Embarked'].replace(['Q','S','C'],['0','1','2'],inplace=True)

#Se cambian los valores de la edad faltantes por la media
print(train["Age"].mean())
print(test["Age"].mean())
media=30
train["Age"]=train["Age"].replace(np.nan,media)
test["Age"]=test["Age"].replace(np.nan,media)

#Se elimia las filas con datos perdidos
train.dropna(axis=0,how='any',inplace=True)
test.dropna(axis=0,how='any',inplace=True)

#Se elimina las columnas PassengerId, Name, Ticket & Cabin
train=train.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
test=test.drop(["Name","Ticket","Cabin"],axis=1)

#Se crean varios grupos segun la edad
#Bandas 0-8,9-15,16-18,19-25,26-40,41-60,61-100
bins=[0,8,15,18,25,40,60,100]
names=['1','2','3','4','5','6','7']
train['Age']=pd.cut(train["Age"],bins,labels=names)
test['Age']=pd.cut(test["Age"],bins,labels=names)

#Verificamos los datos
print(pd.isnull(train).sum())
print(pd.isnull(test).sum())

print(train.shape)
print(train.shape)

print(train.head())
print(test.head())

#|==================================|
#|Implementación de Machine Learning|
#|==================================|

X=np.array(train.drop(['Survived'],1))
y=np.array(train['Survived'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#-------------------
#Regresión Logística
#-------------------
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
Y_pred=logreg.predict(X_test)
print("Presición de la regresión logistica: ",logreg.score(X_train,y_train))

#-------------------------------
#Maquinas de Vectores de Soporte
#-------------------------------
MSV=SVC()
MSV.fit(X_train,y_train)
Y_pred_MSV=MSV.predict(X_test)
print('Presición del algoritmo Maquinas de Vectores de Soporte es: {}'.format(MSV.score(X_train,y_train)))

#--------------------
#Vecinos más Cercanos
#--------------------
VC=KNeighborsClassifier(n_neighbors=3)
VC.fit(X_train,y_train)
Y_pred_VC=VC.predict(X_test)
print('Presición del algoritmo Vecinos más Cercanos es: {}'.format(VC.score(X_train,y_train)))

#-------------------
#Arbol de Decisiones
#-------------------
AD=DecisionTreeClassifier()
AD.fit(X_train,y_train)
Y_pred_AD=AD.predict(X_test)
print('Presición del algoritmo Árbol de Decisiones es: {}'.format(AD.score(X_train,y_train)))

#Ya contamos con los modelos entrenados, ahora usamos el csv de prueba
ids=test['PassengerId']
sex=test['Sex']

#-------------------
#Regresión Logística
#-------------------
prediccion_logreg=logreg.predict(test.drop('PassengerId',axis=1))
out_logreg=pd.DataFrame({'PassengerId':ids,'Survived':prediccion_logreg,'Sex':sex})
out_logreg['Sex'].replace([0,1],['F','M'],inplace=True)
out_logreg['Survived'].replace([0,1],['N','Y'],inplace=True)
print("Predicción de la regresión logistica: \n",out_logreg.head())

#-------------------------------
#Maquinas de Vectores de Soporte
#-------------------------------
prediccion_MSV=MSV.predict(test.drop('PassengerId',axis=1))
out_MSV=pd.DataFrame({'PassengerId':ids,'Survived':prediccion_MSV,'Sex':sex})
out_MSV['Sex'].replace([0,1],['F','M'],inplace=True)
out_MSV['Survived'].replace([0,1],['N','Y'],inplace=True)
print("Predicción de mauina de vectores de soporte: \n",out_MSV.head())

#--------------------
#Vecinos más Cercanos
#--------------------
prediccion_VC=VC.predict(test.drop('PassengerId',axis=1))
out_VC=pd.DataFrame({'PassengerId':ids,'Survived':prediccion_VC,'Sex':sex})
out_VC['Sex'].replace([0,1],['F','M'],inplace=True)
out_VC['Survived'].replace([0,1],['N','Y'],inplace=True)
print("Predicción de Vecinos más Cercanos: \n",out_VC.head())

#-------------------
#Arbol de Decisiones
#-------------------
prediccion_AD=AD.predict(test.drop('PassengerId',axis=1))
out_AD=pd.DataFrame({'PassengerId':ids,'Survived':prediccion_AD,'Sex':sex})
out_AD['Sex'].replace([0,1],['F','M'],inplace=True)
out_AD['Survived'].replace([0,1],['N','Y'],inplace=True)
print("Predicción del Arbol de Decisiones: \n",out_AD.head())

#========
#Graficos
#========

#-------------------------------
#Sobrevivientes respecto al sexo
#-------------------------------

fig,axs=plt.subplots(2,2)
out_logreg.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[0,0])
out_VC.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[0,1])
out_AD.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[1,0])
out_MSV.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[1,1])
for ax in axs.flat:
	ax.set(xlabel='Sexo',ylabel='Numero de personas')
axs[0,0].set_title('Regresión Logística')
axs[0,1].set_title('Maquina de vectores de soporte')
axs[1,0].set_title('Vecinos más cercanos')
axs[1,1].set_title('Árbol de Decisiones')
plt.show()
