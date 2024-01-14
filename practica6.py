import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Crear un conjunto de datos ficticio
data = {
    'ID_cliente': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Edad': [25, 45, 30, 22, 35, 55, 28, 40, 50, 32],
    'Gasto_mensual': [200, 800, 300, 150, 500, 1000, 180, 700, 900, 400],
    'Visitas_mensuales': [4, 12, 6, 3, 8, 15, 5, 10, 14, 7]
}

df = pd.DataFrame(data)

# Mostrar el conjunto de datos
print(df)

# Seleccionar las características relevantes
features = df[['Edad', 'Gasto_mensual', 'Visitas_mensuales']]

# Escalar las características para que tengan media cero y desviación estándar uno
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Aplicar el algoritmo de K-Means para agrupar a los clientes en 3 categorías
kmeans = KMeans(n_clusters=3, random_state=42)
df['Categoria'] = kmeans.fit_predict(scaled_features)

# Mostrar el resultado con las categorías asignadas a cada cliente
print(df)
