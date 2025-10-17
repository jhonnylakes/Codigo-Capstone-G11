import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans

data_adicional_path = "../Datos/additional_data.csv"
data_consumidor_path = "../Datos/consumer_data.csv"
data_proveedor_path = "../Datos/producter_data.csv"

data_adicional = pd.read_csv(data_adicional_path)
data_consumidor = pd.read_csv(data_consumidor_path)
data_proveedor = pd.read_csv(data_proveedor_path)

#Definicion de algunos parametros
clusters = 11
#Usamos semilla, para poder replicar resultados
seed = 42

def parse_location(loc_str):
    if isinstance(loc_str, str):
        return tuple(map(float, ast.literal_eval(loc_str)))
    elif isinstance(loc_str, (tuple, list)):
        return tuple(loc_str)
    else:
        raise ValueError(f"Formato inesperado: {loc_str}")

#se realiza una columna en especifico para las coordenadas x e y
data_consumidor["x"], data_consumidor["y"] = zip(*data_consumidor["Location"].map(parse_location))
data_proveedor["x"], data_proveedor["y"] = zip(*data_proveedor["Location"].map(parse_location))


# Unir consumidores y proveedores en un solo DataFrame
cons = data_consumidor[["Consumer", "x", "y", "Demand", "Ports"]].copy()
cons["Tipo"] = "Consumidor"
cons["Capacity"] = data_consumidor["Capacity"]

prod = data_proveedor[["Producer", "x", "y", "Offer", "Ports"]].copy()
prod["Tipo"] = "Proveedor"
prod["Capacity"] = np.nan

cons.rename(columns={"Consumer": "ID"}, inplace=True)
prod.rename(columns={"Producer": "ID"}, inplace=True)

nodos = pd.concat([cons, prod], ignore_index=True)

# CLUSTERING DE CONSUMIDORES (SOLO BASADO EN DEMANDA)
#Primero lo que hicimos fue hacer un clustering basado en los consumidores, de manera que queden en el 
#mismo cluster los consumidores que estan mas cercanos entre si y por demanda, esto por medio del metodo
# KMeans el cual hace un clustering basado en la distancia euclidiana al centroide del cluster

print("="*80)
print("🎯 FASE 1: CLUSTERING DE CONSUMIDORES")
print("="*80)

X = nodos[["x", "y"]].to_numpy(dtype=float)

def obtener_pesos(df):
    """Pesos solo para consumidores (proveedores = 0)"""
    pesos = []
    for _, row in df.iterrows():
        if row["Tipo"] == "Consumidor":
            peso = row["Demand"] if row["Demand"] > 0 else 1.0
        else:
            peso = 0.0  # Proveedores NO influyen en clustering
        pesos.append(peso)
    return np.array(pesos, dtype=float)

# obtener pesos
weights = obtener_pesos(nodos)

# Clustering basado en consumidores
kmeans = KMeans(n_clusters=clusters, init="k-means++", n_init=100, random_state=seed)
labels = kmeans.fit_predict(X, sample_weight=weights)
nodos["cluster"] = labels
centroides = pd.DataFrame(kmeans.cluster_centers_, columns=["cx", "cy"])
centroides["cluster"] = centroides.index

print(f"✅ {clusters} clusters creados basados en consumidores")

# ASIGNACIÓN OPTIMIZADA DE PROVEEDORES A CLUSTERS
print("\n" + "="*80)
print("🏭 FASE 2: ASIGNACIÓN OPTIMIZADA DE PROVEEDORES")
print("="*80)

# Función para calcular distancia euclidiana
def calcular_distancia(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Separar consumidores y proveedores
consumidores = nodos[nodos['Tipo'] == 'Consumidor'].copy()
proveedores = nodos[nodos['Tipo'] == 'Proveedor'].copy()

# Análisis de clusters (solo consumidores)
print("\n📊 Análisis de clusters de consumidores:")
for cluster_id in sorted(consumidores['cluster'].unique()):
    cons_cluster = consumidores[consumidores['cluster'] == cluster_id]
    demanda_total = cons_cluster['Demand'].sum()
    num_consumidores = len(cons_cluster)
    centroide_x = cons_cluster['x'].mean()
    centroide_y = cons_cluster['y'].mean()
    
    print(f"   Cluster {cluster_id}: {num_consumidores} cons., demanda={demanda_total:.2f}, centroide=({centroide_x:.1f}, {centroide_y:.1f})")

print("Asignacion proveedores a clusters")
# Función optimizada para asignar proveedores a clusters minimizando distancia y balanceando oferta-demanda
def asignar_proveedores_optimizado(nodos_df, num_clusters):
    consumidores_df = nodos_df[nodos_df['Tipo'] == 'Consumidor'].copy()
    proveedores_df = nodos_df[nodos_df['Tipo'] == 'Proveedor'].copy()
    clusters_info = {}
    for cluster_id in range(num_clusters):
        cons_cluster = consumidores_df[consumidores_df['cluster'] == cluster_id]
        
        if len(cons_cluster) > 0:
            clusters_info[cluster_id] = {
                'centroide_x': cons_cluster['x'].mean(),
                'centroide_y': cons_cluster['y'].mean(),
                'demanda_total': cons_cluster['Demand'].sum(),
                'num_consumidores': len(cons_cluster),
                'proveedores_asignados': [],
                'oferta_asignada': 0
            }
    
    # Calcular matriz de distancias: proveedor -> centroide de cluster
    distancias_prov_cluster = {}
    for _, prov in proveedores_df.iterrows():
        prov_id = prov['ID']
        distancias_prov_cluster[prov_id] = {}
        
        for cluster_id, info in clusters_info.items():
            distancia = calcular_distancia(
                prov['x'], prov['y'],
                info['centroide_x'], info['centroide_y']
            )
            distancias_prov_cluster[prov_id][cluster_id] = distancia
    
    # Asegurar al menos un proveedor por cluster

    proveedores_disponibles = set(proveedores_df['ID'])
    for cluster_id in sorted(clusters_info.keys()):
        mejor_proveedor = None
        mejor_distancia = float('inf')
        
        for prov_id in proveedores_disponibles:
            distancia = distancias_prov_cluster[prov_id][cluster_id]
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_proveedor = prov_id
        
        if mejor_proveedor:
            clusters_info[cluster_id]['proveedores_asignados'].append(mejor_proveedor)
            oferta_prov = proveedores_df[proveedores_df['ID'] == mejor_proveedor]['Offer'].iloc[0]
            clusters_info[cluster_id]['oferta_asignada'] += oferta_prov
            proveedores_disponibles.remove(mejor_proveedor)
            print(f"   Cluster {cluster_id} ← Proveedor {mejor_proveedor} (dist: {mejor_distancia:.1f}) con coordenadas ({proveedores_df[proveedores_df['ID'] == mejor_proveedor]['x'].iloc[0]:.1f}, {proveedores_df[proveedores_df['ID'] == mejor_proveedor]['y'].iloc[0]:.1f})")
    
    # Asignar proveedores restantes balanceando oferta-demanda
    print("\n🔄 Asignación secundaria (balancear oferta-demanda):")
    for prov_id in proveedores_disponibles:
        mejor_cluster = None
        mayor_score = -float('inf')
        
        for cluster_id, info in clusters_info.items():
            deficit = info['demanda_total'] - info['oferta_asignada']
            deficit_relativo = deficit / info['demanda_total'] if info['demanda_total'] > 0 else 0
            distancia = distancias_prov_cluster[prov_id][cluster_id]
            score = deficit_relativo - (distancia / 1000)  # Penalizar distancia
            
            if score > mayor_score:
                mayor_score = score
                mejor_cluster = cluster_id
        
        if mejor_cluster is not None:
            clusters_info[mejor_cluster]['proveedores_asignados'].append(prov_id)
            oferta_prov = proveedores_df[proveedores_df['ID'] == prov_id]['Offer'].iloc[0]
            clusters_info[mejor_cluster]['oferta_asignada'] += oferta_prov
            print(f"   Cluster {mejor_cluster} ← Proveedor {prov_id}, dist: {distancias_prov_cluster[prov_id][mejor_cluster]:.1f}, oferta: {oferta_prov:.0f}")
    
    return clusters_info, distancias_prov_cluster

# Ejecutar asignación optimizada
clusters_info, distancias_matriz = asignar_proveedores_optimizado(nodos, clusters)

# Aplicar asignación al DataFrame
nodos.loc[nodos['Tipo'] == 'Proveedor', 'cluster'] = -1

for cluster_id, info in clusters_info.items():
    for prov_id in info['proveedores_asignados']:
        nodos.loc[nodos['ID'] == prov_id, 'cluster'] = cluster_id


# Actualizar referencias
consumidores = nodos[nodos['Tipo'] == 'Consumidor'].copy()
proveedores = nodos[nodos['Tipo'] == 'Proveedor'].copy()