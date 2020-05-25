#-----------------------------------
#Importamos las librerias a utilizar
#-----------------------------------

#Librerias de datascience
import numpy as np
import pandas as pd
#Librerias para gráficar
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import requests
import plotly.express as px
#Librerias de Machine Learning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#---------
#Funciones
#---------

def values(ax,rotacion):
    # Se crea una lista que junta los datos plt.patches
    totals = []
    # Encuentra los valores y los agrega a la lista
    for i in ax.patches:
        totals.append(i.get_height())
    # Se obtiene la suma total de la lista
    total = sum(totals)
    # Se establecen las etiquetas de cada barra con la lista
    for i in ax.patches:
        ax.text(i.get_x()+0.05, i.get_height()+40, \
                str(round((i.get_height()/total)*100, 2))+'% '+str(i.get_height()),
                fontsize=10,color='black',rotation=rotacion)

def autolabel(datos,rotacion):
    for dato in datos:
        height=dato.get_height()
        ax1.annotate('{}'.format(height),
                    xy=(dato.get_x() + dato.get_width() / 2, height),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',rotation=rotacion)

#------------------
#Preparamos la Data
#------------------
#Estos son los colores a utilizar
colors = ['#b00c00', '#edad5f', '#d69e04', '#b5d902', '#63ba00', '#05b08e', '#128ba6',
    '#5f0da6', '#b30bb0', '#c41484', '#a1183d', '#3859eb', '#4da1bf', '#6bcfb6']
colors2=colors[::-1]
colors3=['black','orange','chocolate','tomato']
#Para las graficas circulares se observen mucho mejor
explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

#Se lee el primer archivo que es el de casos confirmados
casos_confirmados=pd.read_csv('casos_confirmados.csv')
print(casos_confirmados.head())

#Se lee el segundo archivo que es el de casos acumulados
casos_acumulados=pd.read_csv('covid_mx.csv')
print(casos_acumulados.head())

#Se lee el tercer archivo que coniene la descripción de todos los pacientes
covid_MX=pd.read_csv('covid-19_general_MX.csv')
print(covid_MX.head())

#Se lee el cuarto archivo que coniene la descripción de todos los sectores
df_sector=pd.read_csv('SECTOR.csv')
print(df_sector.head())

#Se lee el quinto archivo que coniene la descripción de las entidades
df_entidades=pd.read_csv('ENTIDADES.csv')
print(df_entidades.head())

#En el tercer archivo se basara el proceso de machine learning

#Analizamos los datos que tenemos disponibles
print("\nInformación del tipo de dato en el Dataset:")
print(covid_MX.info())

#Describimos la información del Dataset
print("\nDescripción de las estadisticas del Dataset:")
print(covid_MX.describe())

#Verificamos si hay un dato faltante
print("\nDatos faltantes:")
print(pd.isnull(covid_MX).sum())

#----------------------
#Procesamiento de datos
#----------------------

#Se modificaran los valores numericos para un mejor manejo de la información

#Se cambiara el sexo de los pasajeros 0 para mujeres y 1 para hombres
covid_MX['SEXO'].replace([1,2],[0,1],inplace=True)

#Se cambiara el tipo de paciente de 0 para ambulatorio y 1 para hospitalizado
covid_MX['TIPO_PACIENTE'].replace([1,2],[0,1],inplace=True)

#Se modificara el estado de intubado 0 es si, 1 es no, 3 no aplica y 5 no especificado
covid_MX['INTUBADO'].replace([1,2,97,99],[0,1,3,5],inplace=True)

#Se modificara el estado de si el paciente cuenta con neumonia 0 es no, 1 es si y 5 no especificado
covid_MX['NEUMONIA'].replace([1,2,99],[0,1,5],inplace=True)

#Se crean varios grupos segun la edad
#Bandas 0-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100 y 101-115
bins=[-1,10,20,30,40,50,60,70,80,90,100,115]
names1=['0 a 10','11 a 20','21 a 30','31 a 40','41 a 50','51 a 60','61 a 70',
'71 a 80','81 a 90','91 a 100','101 a 115']
names=['1','2','3','4','5','6','7','8','9','10','11']
covid_MX['EDAD']=pd.cut(covid_MX['EDAD'],bins,labels=names)

#Se modificara si cuenta con Diabetes 0 es si, 1 es no y 4 se ignora
covid_MX['DIABETES'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con enfermedad pulmonar obstructiva crónica 0 es si, 1 es no y 4 se ignora
covid_MX['EPOC'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con asma 0 es si, 1 es no y 4 se ignora
covid_MX['ASMA'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con inmusupr 0 es si, 1 es no y 4 se ignora
covid_MX['INMUSUPR'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con hipertensión 0 es si, 1 es no y 4 se ignora
covid_MX['HIPERTENSION'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con otra con 0 es si, 1 es no y 4 se ignora
covid_MX['OTRA_CON'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con enfermedad cardiovascular 0 es si, 1 es no y 4 se ignora
covid_MX['CARDIOVASCULAR'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con obesidad 0 es si, 1 es no y 4 se ignora
covid_MX['OBESIDAD'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con enfermedad renal crónica 0 es si, 1 es no y 4 se ignora
covid_MX['RENAL_CRONICA'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con tabaquismo 0 es si, 1 es no y 4 se ignora
covid_MX['TABAQUISMO'].replace([1,2,98],[0,1,4],inplace=True)

#Se modificara si cuenta con otro caso 0 es si, 1 es no y 4 se ignora
covid_MX['OTRO_CASO'].replace([1,2,99],[0,1,5],inplace=True)

#Se modificara si cuenta con unidad de cuidados intensivos 0 es si, 1 es no, 3 no aplica y 5 no especificado
covid_MX['UCI'].replace([1,2,97,99],[0,1,3,5],inplace=True)

#Se modificara si cuenta con otro caso 0 es positivo, 1 es negativo y 2 pendiente
covid_MX['RESULTADO'].replace([1,2,3],[0,1,2],inplace=True)

#Se transforma de objeto fecha de ingreso a fecha
covid_MX["FECHA_INGRESO"] = pd.to_datetime(covid_MX["FECHA_INGRESO"])

#Se transforma de objeto fecha de sintomas a fecha
covid_MX["FECHA_SINTOMAS"] = pd.to_datetime(covid_MX["FECHA_SINTOMAS"])

#Se modifican los valores nulos por una fecha lejana
covid_MX["FECHA_DEF"]=covid_MX["FECHA_DEF"].replace('9999-99-99','31/12/2020')

#Se transforma de objeto fecha de defuncion a fecha
covid_MX["FECHA_DEF"] = pd.to_datetime(covid_MX["FECHA_DEF"])

#Corrigiendo la información de casos confirmados para el mapa de México
estados_or=['Ciudad De México','Veracruz De Ignacio De La Llave','Michoacán De Ocampo','Coahuila De Zaragoza']
estados_or2=['Ciudad de México','Veracruz','Michoacán','Coahuila']
estados=np.array(list(df_entidades['ENTIDAD_FEDERATIVA']))
df_entidades['ENTIDAD_FEDERATIVA']=[i.title() for i in estados]
df_entidades['ENTIDAD_FEDERATIVA'].replace(estados_or,estados_or2,inplace=True)

#Ahora eliminaremos la primer columa que no tiene nombre
covid_MX=covid_MX.drop(covid_MX.columns[0],axis=1)

print(covid_MX.head())

print(covid_MX.shape)

print(covid_MX.info())

print(pd.isnull(covid_MX).sum())

#----------------------
#Visualización de datos
#----------------------

#Número de infectados por día en México
#-----------------------------------------

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
casos_acumulados['Cases per Day']=[len(covid_MX.loc[(covid_MX['FECHA_INGRESO']== n) & 
    (covid_MX['RESULTADO']==0)]) for n in list(casos_acumulados['Dates'])]
labels=casos_acumulados['Dates']
x=np.arange(len(labels))
y1=np.array(casos_acumulados['Cases per Day'])
z=np.polyfit(x,y1,6)
p=np.poly1d(z)

#Aquí se genera el gráfico
datos1=ax1.bar(x,casos_acumulados['Cases per Day'],color='black')
ax1.plot(x,casos_acumulados['Cases per Day'],'.-',color='orange')
ax1.plot(x,p(x),'--r',label='Linea de tendencia')
ax1.set_title('Número de infectados por día en México')
ax1.set_xlabel('Día')
ax1.set_ylabel('Numero de infectados')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=90)
autolabel(datos1,90)
plt.show()

#Número de fallecidos por día en México
#--------------------------------------

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
casos_acumulados['Deaths per Day']=[len(covid_MX.loc[(covid_MX['FECHA_DEF']== n) & 
    (covid_MX['RESULTADO']==0)]) for n in list(casos_acumulados['Dates'])]
labels=casos_acumulados['Dates']
x=np.arange(len(labels))
y1=np.array(casos_acumulados['Deaths per Day'])
z=np.polyfit(x,y1,6)
p=np.poly1d(z)

#Aquí se genera el gráfico
datos1=ax1.bar(x,casos_acumulados['Deaths per Day'],color='black')
ax1.plot(x,casos_acumulados['Deaths per Day'],'.-',color='orange')
ax1.plot(x,p(x),'--r',label='Linea de tendencia')
ax1.set_title('Número de fallecidos por día en México')
ax1.set_xlabel('Día')
ax1.set_ylabel('Numero de fallecidos')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=90)
autolabel(datos1,90)
plt.show()

#Número de infectados acumulados en México
#-----------------------------------------

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
labels=casos_acumulados['Dates']
x=np.arange(len(labels))
datos1=ax1.bar(x,casos_acumulados['Confirmed Cases'],color='black')
ax1.plot(x,casos_acumulados['Confirmed Cases'],'.-',color='orange')
ax1.set_title('Número de infectados acumulados en México')
ax1.set_xlabel('Día')
ax1.set_ylabel('Numero de infectados')
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=90)
autolabel(datos1,90)
plt.show()

#Número de fallecidos acumulados en México
#-----------------------------------------

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
labels=casos_acumulados['Dates']
x=np.arange(len(labels))
datos1=ax1.bar(x,casos_acumulados['Deceased'],color='black')
ax1.plot(x,casos_acumulados['Deceased'],'.-',color='orange')
ax1.set_title('Número de fallecidos acumulados en México')
ax1.set_xlabel('Día')
ax1.set_ylabel('Numero de fallecidos')
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=90)
autolabel(datos1,90)
plt.show()

#Número de infectados en México por estado
#-----------------------------------------

#Se contabiliza el numero de fallecidos por cada institución de salud
df_entidades['CASOS']=[len(covid_MX.loc[(covid_MX['ENTIDAD_UM']== n) &
		(covid_MX['RESULTADO']==0)]) for n in list(df_entidades['CLAVE_ENTIDAD'])]

repo_url = 'https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json' #Archivo GeoJSON
mx_regions_geo = requests.get(repo_url).json()

fig = px.choropleth(data_frame=df_entidades, 
                    geojson=mx_regions_geo, #Se hace referencia a la linea mx_regions_geo
                    locations='ENTIDAD_FEDERATIVA', # nombre de la columna del Dataframe
                    featureidkey='properties.name',  # ruta al campo del archivo GeoJSON con el que se hará la relación (nombre de los estados)
                    color='CASOS', #El color depende de las cantidades
                    color_continuous_scale="Reds",
                   )
fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

fig.update_layout(
    title_text = 'Casos de infección en México',
    font=dict(
        family="Ubuntu",
        size=18,
        color="#7f7f7f"
    ),
    annotations = [dict(
        x=0.55,
        y=-0.1,
        xref='paper',
        yref='paper',
        showarrow = False
    )]
)
fig.show()
plt.show()

#Covid-19 casos confirmados segun su estado
#------------------------------------------

#Personas cuyo resultado fue positivo
Covid_positivo=covid_MX.loc[covid_MX['RESULTADO']==0]
#Personas fallecidas con resultado positivo
Covid_muerto_positivo=covid_MX.loc[(covid_MX['RESULTADO']==0) &
	(covid_MX['FECHA_DEF']!='31/12/2020') & (covid_MX['FECHA_DEF'].notnull())]
#Personas vivas con resultado positivo pero intubada
Vivo_intubado=covid_MX.loc[(covid_MX['RESULTADO']==0) & ((covid_MX['FECHA_DEF']=='31/12/2020') |
	(covid_MX['FECHA_DEF'].notnull()))&(covid_MX['INTUBADO']==0)]
#Personas vivas con resultado positivo en estado crítico
Vivo_ICU=covid_MX.loc[(covid_MX['RESULTADO']==0) & ((covid_MX['FECHA_DEF']=='31/12/2020') |
	(covid_MX['FECHA_DEF'].notnull())) & (covid_MX['UCI']==0)]

#Se obtiene el numero total del estado de las personas
cptotal=len(Covid_positivo)
cmptotal=len(Covid_muerto_positivo)
Vitotal=len(Vivo_intubado)
Vicutotal=len(Vivo_ICU)

#Se realiza las características para crear el gráfico
sizes=np.array([cptotal-cmptotal-Vitotal-Vicutotal,cmptotal,Vitotal,Vicutotal])
titulos=['Positivos','Fallecidos','Vivo intubado','Vivo cuidado intensivo']
porcentaje=100.*sizes/sizes.sum()
leyenda=['{0} - {1:0.2f}% = {2:0.0f}'.format(titulos[i],porcentaje[i],sizes[i]) for i in range(len(titulos))]

#Se crea el gráfico
fig, ax1=plt.subplots(figsize=(10,6))
ax1.pie(sizes,startangle=90,shadow=True,explode=(0.1,0.1,0.1,0.1),colors=colors)
ax1.set_title('Distribución de los casos confirmados')
ax1.legend(leyenda,loc='best',fontsize=10,bbox_to_anchor=(-0.1, 1.))
fig.tight_layout()
plt.show()

#Personas fallecidas con Covid-19 distribuidas según la institución de salud
#---------------------------------------------------------------------------

#Se obtiene el total de casos positivos fallecidos
df_sector['TOTAL_POSITIVOS_DEF']=[len(covid_MX.loc[(covid_MX['SECTOR']==n) &
	(covid_MX['FECHA_DEF']!='31/12/2020') & (covid_MX['FECHA_DEF'].notnull()) &
	(covid_MX['RESULTADO']==0)])for n in list(df_sector['CLAVE'])]
#Se acomodan los valores de manera descendente
df_sector=df_sector.sort_values('TOTAL_POSITIVOS_DEF',ascending=False)

#Se realiza las características para crear el gráfico
tamaño=np.array(list(df_sector['TOTAL_POSITIVOS_DEF']))
titulos=[i for i in df_sector['DESCRIPCIÓN']]
porcentaje=100.0*tamaño/tamaño.sum()
leyenda=['{0} - {1:0.2f}% = {2:0.0f}'.format(titulos[i],porcentaje[i],tamaño[i]) for i in range(len(titulos))]

#Se crea el gráfico
fig, ax1=plt.subplots(figsize=(10,6))
ax1.pie(tamaño,startangle=90,shadow=True,colors=colors,explode=explode)
ax1.set_title('Personas fallecidas con Covid-19 distribuidas según la institución de salud')
ax1.legend(leyenda,loc='best',fontsize=10,bbox_to_anchor=(-0.1, 1.))
fig.tight_layout()
plt.show()

#Casos confirmados positivos y fallecidos según la institución de salud
#----------------------------------------------------------------------

#Se contabiliza el numero de fallecidos por cada institución de salud
df_sector['TOTAL_DEF']=[len(covid_MX.loc[(covid_MX['SECTOR']== n) & (covid_MX['FECHA_DEF']!='31/12/2020')
	& (covid_MX.FECHA_DEF.notnull())]) for n in list(df_sector['CLAVE'])]
#Se contabiliza el numero de personas con covid
df_sector['TOTAL_POS']=[len(covid_MX.loc[(covid_MX['SECTOR']== n) & (covid_MX['RESULTADO']==0)])
	for n in list(df_sector['CLAVE'])]
#Se acomodan los valores de manera descendente
df_sector=df_sector.sort_values('TOTAL_POS',ascending=False)

#Se realiza las características para crear el gráfico
total_positivos=np.array(list(df_sector['TOTAL_POS'])) #Numero de casos positivos
total_positivos_muertos=np.array(list(df_sector['TOTAL_POSITIVOS_DEF'])) #Numero de muertos positivos
porcentaje1=100.0*total_positivos_muertos/total_positivos.sum() #La tasa de Mortalidad

#Se crea el gráfico
fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
x=np.arange(len(df_sector['DESCRIPCIÓN'])) #Los titulos en el eje X
w=0.6 #Ancho de la gráfica
graf1=ax1.bar(x,total_positivos,width=w/2,align='center') #Grafica de casos positivos
graf2=ax1.bar(x+w,total_positivos_muertos,width=w/2,align='center') #Grafica de casos positivos muertos
#Se modifican los colores para cada barra
for i, bar in enumerate(graf1):
    bar.set_color(colors[i])
for i, bar in enumerate(graf2):
    bar.set_color(colors2[i])
#Se modifican los labels en X
plt.xticks(x + w /2, df_sector['DESCRIPCIÓN'], rotation='vertical',fontsize=7)
#Se prepara la leyenda según el color de cada grafico
ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], 
	label='{0} - {1:0.2f}%'.format(list(df_sector['DESCRIPCIÓN'])[x],porcentaje1[x]))
    for x in range(len(df_sector['DESCRIPCIÓN']))], 
	loc='best',bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True, title="Tasa de Letalidad")
ax1.set_title('Casos confirmados positivos(total) y fallecidos con covid')
plt.show()

"""
Esto es para graficar con SEABORNE
g1=sns.barplot(x='DESCRIPCIÓN', y='TOTAL_POS',data=df_sector,ax=ax1,hue=leyenda1)
g2=sns.barplot(x='DESCRIPCIÓN', y='TOTAL_POSITIVOS_DEF', data=df_sector,ax=ax2)
g1.set_xticklabels(labels, rotation='vertical',fontsize=5)
g2.set_xticklabels(labels, rotation='vertical',fontsize=5)
"""

#----------------
#Machine Learning
#----------------

#Creamos la columa de sobrevivio ante la enfermedad 0 es si y 1 es no
#Condición uno si ha muerto y el resultado es positivo se le pone 1, si no es 0
condiciones=[(covid_MX['FECHA_DEF']=='31/12/2020') & (covid_MX['RESULTADO']==0),
    (covid_MX['FECHA_DEF']!='31/12/2020') & (covid_MX['RESULTADO']==0)]
resultados=[0,1]
covid_MX['SOBREVIVIO']=np.select(condiciones,resultados,default=np.nan)
#Quitamos todas las filas vacias
covid_MX.dropna(axis=0, how='any', inplace=True)

No_necesario=['SECTOR','ENTIDAD_UM','ENTIDAD_RES','FECHA_INGRESO','FECHA_SINTOMAS',
    'FECHA_DEF','NACIONALIDAD','SOBREVIVIO']

y=np.array(covid_MX['SOBREVIVIO'])
X=np.array(covid_MX.drop(No_necesario,1))

#Creamos los arrays de prueba y test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#-------------------
#Regresión Logística
#-------------------
logreg=LogisticRegression(max_iter=300)
logreg.fit(X_train,y_train)
Y_pred=logreg.predict(X_test)

#-------------------------------
#Maquinas de Vectores de Soporte
#-------------------------------
MSV=SVC(max_iter=300)
MSV.fit(X_train,y_train)
Y_pred_MSV=MSV.predict(X_test)

#-------------------
#Arbol de Decisiones
#-------------------
AD=DecisionTreeClassifier()
AD.fit(X_train,y_train)
Y_pred_AD=AD.predict(X_test)

#Presición de los modelos
print("Presición de la regresión logistica: ",logreg.score(X_train,y_train))
print('Presición del algoritmo Maquinas de Vectores de Soporte es: {}'.format(MSV.score(X_train,y_train)))
print('Presición del algoritmo Árbol de Decisiones es: {}'.format(AD.score(X_train,y_train)))

#Predicción de los modelos
#-------------------------

Sexo=covid_MX['SEXO']
Sobrev=covid_MX['SOBREVIVIO']
original=pd.DataFrame({'Survived':Sobrev,'Sex':Sexo})
original['Sex'].replace([0,1],['M','H'],inplace=True)
original['Survived'].replace([0,1],['Vivo','Muerto'],inplace=True)

#Regresión Logística
#-------------------
prediccion_logreg=logreg.predict(covid_MX.drop(No_necesario,1))
out_logreg=pd.DataFrame({'Survived':prediccion_logreg,'Sex':Sexo})
out_logreg['Sex'].replace([0,1],['M','H'],inplace=True)
out_logreg['Survived'].replace([0,1],['Vivo','Muerto'],inplace=True)
print("Predicción de la regresión logistica: \n",out_logreg.head())

#Maquinas de Vectores de Soporte
#-------------------------------
prediccion_MSV=MSV.predict(covid_MX.drop(No_necesario,1))
out_MSV=pd.DataFrame({'Survived':prediccion_MSV,'Sex':Sexo})
out_MSV['Sex'].replace([0,1],['M','H'],inplace=True)
out_MSV['Survived'].replace([0,1],['Vivo','Muerto'],inplace=True)
print("Predicción de mauina de vectores de soporte: \n",out_MSV.head())

#Arbol de Decisiones
#-------------------
prediccion_AD=AD.predict(covid_MX.drop(No_necesario,1))
out_AD=pd.DataFrame({'Survived':prediccion_AD,'Sex':Sexo})
out_AD['Sex'].replace([0,1],['M','H'],inplace=True)
out_AD['Survived'].replace([0,1],['Vivo','Muerto'],inplace=True)
print("Predicción del Arbol de Decisiones: \n",out_AD.head())

fig,axs=plt.subplots(2,2)
graf1=original.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[0,0],color=colors3)
graf2=out_logreg.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[0,1],color=colors3)
graf3=out_AD.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[1,0],color=colors3)
graf4=out_MSV.groupby(['Sex','Survived']).size().unstack().plot(kind='bar',ax=axs[1,1],color=colors3)
values(graf1,0)
values(graf2,0)
values(graf3,0)
values(graf4,0)
for ax in axs.flat:
    ax.set(xlabel='Sexo',ylabel='Numero de personas')
    ax.legend(title='Estado',loc='lower center')
axs[0,0].set_title('Datos Originales')
axs[0,1].set_title('Regresión Logística')
axs[1,0].set_title('Árbol de Decisiones')
axs[1,1].set_title('Maquinas de Vectores de Soporte')
plt.show()

#Se calcula la matriz de confusión
print('Regresión Logística su matriz de confusión {}'.format(confusion_matrix(y,prediccion_logreg)))
print('\nMaquinas de Vectores de Soporte su matriz de confusión {}'.format(confusion_matrix(y,prediccion_MSV)))
print('\nArbol de Decisiones su matriz de confusión {}'.format(confusion_matrix(y,prediccion_AD)))

#--------------------------------------
#Visualización de datos de enfermedades
#--------------------------------------

#Exportamos el archivo con los datos que se obtienen
df_nuevo=covid_MX

#Cambiamos los grupos por el rango de edad
df_nuevo['EDAD'].replace(names,names1,inplace=True)

#Personas con covid según su Sexo
#--------------------------------

condiciones=[(df_nuevo['SEXO']==0) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['SEXO']==0) & (df_nuevo['SOBREVIVIO']==1),
    (df_nuevo['SEXO']==1) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['SEXO']==1) & (df_nuevo['SOBREVIVIO']==1)]
resultados=[0,1,2,3]
df_nuevo['SEXO_C']=np.select(condiciones,resultados,default=np.nan)
df_nuevo['SEXO_C'].replace([0,1,2,3],['Mujeres V','Mujeres F','Hombres V','Hombres F'],inplace=True)
hola=df_nuevo.groupby(['EDAD','SEXO_C']).size().unstack().plot(kind='bar',color=colors3)
values(hola,90)
plt.xlabel('Edad de las personas [Años]')
plt.ylabel('Numero de personas')
plt.legend(title='Estado')
plt.title('Personas con Covid-19 según su sexo',loc='left')
plt.show()

#Personas con enfermedades crónicas
#----------------------------------
neumonia=len(df_nuevo.loc[df_nuevo['NEUMONIA']==0])
diabetes=len(df_nuevo.loc[df_nuevo['DIABETES']==0])
epoc=len(df_nuevo.loc[df_nuevo['EPOC']==0])
asma=len(df_nuevo.loc[df_nuevo['ASMA']==0])
inmunosupresion=len(df_nuevo.loc[df_nuevo['INMUSUPR']==0])
hipertension=len(df_nuevo.loc[df_nuevo['HIPERTENSION']==0])
cardio=len(df_nuevo.loc[df_nuevo['CARDIOVASCULAR']==0])
obesidad=len(df_nuevo.loc[df_nuevo['OBESIDAD']==0])
renal=len(df_nuevo.loc[df_nuevo['RENAL_CRONICA']==0])
tabaquismo=len(df_nuevo.loc[df_nuevo['TABAQUISMO']==0])

textos_enfermedades=['Neumonia','Diabetes','Epoc','Asma','Inmunosupresión',
    'Hipertensión','Cardiovascular','Obesidad','Cronica renal','Tabaquismo']
total=len(df_nuevo['EDAD'])
enfermedades=np.array([neumonia,diabetes,epoc,asma,inmunosupresion,
    hipertension,cardio,obesidad,renal,tabaquismo])
porcentaje1=enfermedades/total*100

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
labels=textos_enfermedades
x=np.arange(len(labels))
datos1=ax1.bar(x,enfermedades,color='black')
for i, bar in enumerate(datos1):
    bar.set_color(colors[i])
#Se prepara la leyenda según el color de cada grafico
ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], 
    label='{0} - {1:0.2f}%'.format(textos_enfermedades[x],porcentaje1[x]))
    for x in range(len(textos_enfermedades))], 
    loc='best', fancybox=True, shadow=True, title="Tasa de contagiados")
ax1.set_title('Personas con enfermedades crónicas y Covid-19 en México')
ax1.set_xlabel('Enfermedades')
ax1.set_ylabel('Personas enfermas')
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=20,fontsize=8)
autolabel(datos1,0)
plt.show()

#Personas fallecidas con enfermedades crónicas
#----------------------------------
neumonia=len(df_nuevo.loc[(df_nuevo['NEUMONIA']==0)&(df_nuevo['SOBREVIVIO']==1)])
diabetes=len(df_nuevo.loc[(df_nuevo['DIABETES']==0)&(df_nuevo['SOBREVIVIO']==1)])
epoc=len(df_nuevo.loc[(df_nuevo['EPOC']==0)&(df_nuevo['SOBREVIVIO']==1)])
asma=len(df_nuevo.loc[(df_nuevo['ASMA']==0)&(df_nuevo['SOBREVIVIO']==1)])
inmunosupresion=len(df_nuevo.loc[(df_nuevo['INMUSUPR']==0)&(df_nuevo['SOBREVIVIO']==1)])
hipertension=len(df_nuevo.loc[(df_nuevo['HIPERTENSION']==0)&(df_nuevo['SOBREVIVIO']==1)])
cardio=len(df_nuevo.loc[(df_nuevo['CARDIOVASCULAR']==0)&(df_nuevo['SOBREVIVIO']==1)])
obesidad=len(df_nuevo.loc[(df_nuevo['OBESIDAD']==0)&(df_nuevo['SOBREVIVIO']==1)])
renal=len(df_nuevo.loc[(df_nuevo['RENAL_CRONICA']==0)&(df_nuevo['SOBREVIVIO']==1)])
tabaquismo=len(df_nuevo.loc[(df_nuevo['TABAQUISMO']==0)&(df_nuevo['SOBREVIVIO']==1)])

enfermedades=np.array([neumonia,diabetes,epoc,asma,inmunosupresion,
    hipertension,cardio,obesidad,renal,tabaquismo])
porcentaje1=enfermedades/total*100

fig, ax1=plt.subplots(figsize=(10,6))
loc,labels=plt.xticks() #Modificar los labels en x
labels=textos_enfermedades
x=np.arange(len(labels))
datos1=ax1.bar(x,enfermedades,color='black')
for i, bar in enumerate(datos1):
    bar.set_color(colors[i])
#Se prepara la leyenda según el color de cada grafico
ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], 
    label='{0} - {1:0.2f}%'.format(textos_enfermedades[x],porcentaje1[x]))
    for x in range(len(textos_enfermedades))], 
    loc='best', fancybox=True, shadow=True, title="Tasa de letalidad")
ax1.set_title('Personas fallecidas con enfermedades crónicas y Covid-19 en México')
ax1.set_xlabel('Enfermedades')
ax1.set_ylabel('Personas fallecidas')
ax1.set_xticks(x)
ax1.set_xticklabels(labels,rotation=20,fontsize=8)
autolabel(datos1,0)
plt.show()

#Personas con diabetes
#---------------------

condiciones=[(df_nuevo['DIABETES']==0) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['DIABETES']==0) & (df_nuevo['SOBREVIVIO']==1)]
resultados=[0,1]
#Se aplicán las condiciones
df_nuevo['DIABETES_C']=np.select(condiciones,resultados,default=np.nan)
#Se reemplazan 0 y 1 con vivo o muerto
df_nuevo['DIABETES_C'].replace([0,1],['Vivo','Muerto'],inplace=True)

hola=df_nuevo.groupby(['EDAD','DIABETES_C']).size().unstack().plot(kind='bar',color=colors3)
values(hola,90)
plt.xlabel('Edad de las personas [Años]')
plt.ylabel('Numero de personas')
plt.legend(title='Estado')
plt.title('Personas Diabeticas con Covid-19',loc='left')
plt.show()

#Personas con hipertensión
#-------------------------

condiciones=[(df_nuevo['HIPERTENSION']==0) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['HIPERTENSION']==0) & (df_nuevo['SOBREVIVIO']==1)]
#Se aplicán las condiciones
df_nuevo['HIPERTENSION_C']=np.select(condiciones,resultados,default=np.nan)

#Se reemplazan 0 y 1 con vivo o muerto
df_nuevo['HIPERTENSION_C'].replace([0,1],['Vivo','Muerto'],inplace=True)

hola=df_nuevo.groupby(['EDAD','HIPERTENSION_C']).size().unstack().plot(kind='bar',color=colors3)
values(hola,90)
plt.xlabel('Edad de las personas [Años]')
plt.ylabel('Numero de personas')
plt.legend(title='Estado')
plt.title('Personas Hipertensas con Covid-19',loc='left')
plt.show()

#Personas con obesidad
#---------------------

condiciones=[(df_nuevo['OBESIDAD']==0) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['OBESIDAD']==0) & (df_nuevo['SOBREVIVIO']==1)]
#Se aplicán las condiciones
df_nuevo['OBESIDAD_C']=np.select(condiciones,resultados,default=np.nan)

#Se reemplazan 0 y 1 con vivo o muerto
df_nuevo['OBESIDAD_C'].replace([0,1],['Vivo','Muerto'],inplace=True)

hola=df_nuevo.groupby(['EDAD','OBESIDAD_C']).size().unstack().plot(kind='bar',color=colors3)
values(hola,90)
plt.xlabel('Edad de las personas [Años]')
plt.ylabel('Numero de personas')
plt.legend(title='Estado')
plt.title('Personas Obesas con Covid-19',loc='left')
plt.show()

#Personas con tabaquismo
#-----------------------

condiciones=[(df_nuevo['TABAQUISMO']==0) & (df_nuevo['SOBREVIVIO']==0),
    (df_nuevo['TABAQUISMO']==0) & (df_nuevo['SOBREVIVIO']==1)]
#Se aplicán las condiciones
df_nuevo['TABAQUISMO_C']=np.select(condiciones,resultados,default=np.nan)

#Se reemplazan 0 y 1 con vivo o muerto
df_nuevo['TABAQUISMO_C'].replace([0,1],['Vivo','Muerto'],inplace=True)

hola=df_nuevo.groupby(['EDAD','TABAQUISMO_C']).size().unstack().plot(kind='bar',color=colors3)
values(hola,90)
plt.xlabel('Edad de las personas [Años]')
plt.ylabel('Numero de personas')
plt.legend(title='Estado')
plt.title('Personas con Tabaquismo y Covid-19',loc='left')
plt.show()

df_nuevo.to_csv('nuevo.csv')