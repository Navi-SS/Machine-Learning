import pandas as pd
import requests
import plotly.express as px
import matplotlib.pyplot as plt
df=pd.read_csv('MexicoData.csv')
#https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json
repo_url = 'https://raw.githubusercontent.com/Navi-SS/Machine-Learning/master/mexicoHigh.json?token=AOGQ4POMOE73ET5XJ32CTPK6XBJGY' #Archivo GeoJSON
mx_regions_geo = requests.get(repo_url).json()

fig = px.choropleth(data_frame=df, 
                    geojson=mx_regions_geo, #Se hace referencia a la linea mx_regions_geo
                    locations='Estado', # nombre de la columna del Dataframe
                    featureidkey='properties.name',  # ruta al campo del archivo GeoJSON con el que se hará la relación (nombre de los estados)
                    color='Casos', #El color depende de las cantidades
                    color_continuous_scale="Purples", #greens
                    #scope="north america"
                   )
fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

fig.update_layout(
    title_text = 'Casos de infección en México',
    font=dict(
        #family="Courier New, monospace",
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