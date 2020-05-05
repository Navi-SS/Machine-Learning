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

iris=pd.read_csv('Iris.csv')
print(iris.head())

#Se elimina la columna Id
iris=iris.drop("Id",axis=1)
print(iris.head())

#Analizamos los datos que tenemos disponibles
print("\nInformación del Dataset: ")
print(iris.info())

#Describimos la información del Dataset
print("\nDescripción del Dataset: ")
print(iris.describe())

#Se verifica la distribución de los datos segun la especie
print("\nDistribución de las especies de Iris: ")
print(iris.groupby('Species').size())

#|====================================================|
#|Preparar la data para el proceso de Machine Learning|
#|====================================================|

def grafico():
	#Graficamos los datos del Sepal
	#(tipo de grafico,datos en eje x,datos en eje y, color del grafico,leyenda)
	fig,(ax1,ax2)=plt.subplots(1,2)
	iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue',label='Setosa',ax=ax1)
	iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Versicolor',ax=ax1)
	iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Virginica',ax=ax1)
	ax1.set_xlabel('Sepalo-Longitud')
	ax1.set_ylabel('Sepalo-Ancho')
	ax1.set_title('Sepalo-Longitud vs Ancho')

	#Graficamos los datos del Petal
	#(tipo de grafico,datos en eje x,datos en eje y, color del grafico,leyenda)
	iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue',label='Setosa',ax=ax2)
	iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='Versicolor',ax=ax2)
	iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='Virginica',ax=ax2)
	ax2.set_xlabel('Petal-Longitud')
	ax2.set_ylabel('Petal-Ancho')
	ax2.set_title('Petal-Longitud vs Ancho')
	plt.show()

def grafico1():
	#sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)
	grid=sns.JointGrid(x='SepalLengthCm',y='SepalWidthCm',data=iris)
	g=grid.plot_joint(sns.scatterplot,hue='Species',data=iris)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-setosa', 'SepalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-versicolor', 'SepalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-virginica', 'SepalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-setosa', 'SepalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-versicolor', 'SepalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-virginica', 'SepalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	plt.show()

	#sns.jointplot(x='PetalLengthCm',y='PetalWidthCm',data=iris)
	grid=sns.JointGrid(x='PetalLengthCm',y='PetalWidthCm',data=iris)
	g=grid.plot_joint(sns.scatterplot,hue='Species',data=iris)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-setosa', 'PetalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-versicolor', 'PetalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-virginica', 'PetalLengthCm'], ax=g.ax_marg_x, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-setosa', 'PetalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-versicolor', 'PetalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	sns.kdeplot(iris.loc[iris['Species']=='Iris-virginica', 'PetalWidthCm'], ax=g.ax_marg_y, vertical=True, legend=False)
	plt.show()

grafico()
grafico1()
#|==================================|
#|Implementación de Machine Learning|
#|==================================|

#Declaramos los valores X y Y
X=np.array(iris.drop(['Species'],1))
Y=np.array(iris['Species'])
#Separamos los datos de train y prueba para los algoritmos
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para la prueba'.format(X_train.shape[0],X_test.shape[0]))

#-------------------
#Regresión Logística
#-------------------
RL=LogisticRegression()
RL.fit(X_train,y_train)
Y_pred_RL=RL.predict(X_test)
print('Precisión del algoritmo Regresión Logística es: {}'.format(RL.score(X_train,y_train)))

#-------------------------------
#Máquinas de Vectores de Soporte
#-------------------------------
MSV=SVC()
MSV.fit(X_train,y_train)
Y_pred_MSV=MSV.predict(X_test)
print('Precisión del algoritmo Máquinas de Vectores de Soporte es: {}'.format(MSV.score(X_train,y_train)))

#--------------------
#Vecinos más Cercanos
#--------------------
VC=KNeighborsClassifier(n_neighbors=10)
VC.fit(X_train,y_train)
Y_pred_VC=VC.predict(X_test)
print('Precisión del algoritmo Vecinos más Cercanos es: {}'.format(VC.score(X_train,y_train)))

#-------------------
#Arbol de Decisiones
#-------------------
AD=DecisionTreeClassifier()
AD.fit(X_train,y_train)
Y_pred_AD=AD.predict(X_test)
print('Precisión del algoritmo Árbol de Decisiones es: {}'.format(AD.score(X_train,y_train)))

grafico()