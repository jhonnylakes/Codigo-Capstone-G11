import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans

# Cargar datos
data_adicional_path = "../Datos/additional_data.csv"
data_consumidor_path = "../Datos/consumer_data.csv"
data_proveedor_path = "../Datos/producter_data.csv"

data_adicional = pd.read_csv(data_adicional_path)
data_consumidor = pd.read_csv(data_consumidor_path)
data_proveedor = pd.read_csv(data_proveedor_path)

clusters = 3
seed = 42
HORIZONTE_TOTAL = 52
VENTANA_ROLLING = 6

def parse_location(loc_str):
    if isinstance(loc_str, str):
        return tuple(map(float, ast.literal_eval(loc_str)))
    elif isinstance(loc_str, (tuple, list)):
        return tuple(loc_str)
    else:
        raise ValueError(f"Formato inesperado: {loc_str}")

data_consumidor["x"], data_consumidor["y"] = zip(*data_consumidor["Location"].map(parse_location))
data_proveedor["x"], data_proveedor["y"] = zip(*data_proveedor["Location"].map(parse_location))


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

# Mostrar resumen
print("\n" + "="*80)
print("📊 RESUMEN DE ASIGNACIÓN")
print("="*80)

for cluster_id in sorted(clusters_info.keys()):
    info = clusters_info[cluster_id]
    balance = info['oferta_asignada'] - info['demanda_total']
    cobertura = (info['oferta_asignada'] / info['demanda_total'] * 100) if info['demanda_total'] > 0 else 0
    
    print(f"\n📍 Cluster {cluster_id}:")
    print(f"   Consumidores: {info['num_consumidores']}, Demanda: {info['demanda_total']:.0f}")
    print(f"   Proveedores: {len(info['proveedores_asignados'])} → {info['proveedores_asignados']}")
    print(f"   Oferta: {info['oferta_asignada']:.0f}, Balance: {balance:+.0f} {'✅' if balance >= 0 else '⚠️'}, Cobertura: {cobertura:.0f}%")

# CONFIGURACIÓN DE SIMULACIÓN
print("\n" + "="*80)
print("⚙️  FASE 3: CONFIGURACIÓN DE SIMULACIÓN")
print("="*80)

num_barcos = clusters
capacidad_barco = data_adicional['Ship_Capacity'].iloc[0]
costo_inventario = data_adicional['Inventory_Cost'].iloc[0]

# Parámetros de política (s, S)
consumidores_info = data_consumidor[['Consumer', 'Demand', 'Capacity', 'Ports']].copy()
consumidores_info['Demanda_Semanal'] = consumidores_info['Demand']
consumidores_info['s_calculado'] = consumidores_info['Demanda_Semanal']
consumidores_info['S_ideal'] = 3 * consumidores_info['Demanda_Semanal']
consumidores_info['S_final'] = consumidores_info[['S_ideal', 'Capacity']].min(axis=1)

demanda_semanal_total = consumidores_info['Demanda_Semanal'].sum()
oferta_semanal_total = proveedores['Offer'].sum()

print(f"\n🚢 Flota: {num_barcos} barcos, capacidad {capacidad_barco:.0f} unidades/barco")
print(f"📦 Política (s,S): s = demanda semanal, S = 1× demanda")
print(f"📊 Demanda total: {demanda_semanal_total:.0f} unidades/semana")
print(f"🏭 Oferta total: {oferta_semanal_total:.0f} unidades/semana")
print(f"📍 Clusters: {clusters}")
print(f"💰 Costo inventario: ${costo_inventario} por unidad")

# Validación: Barcos vs Clusters
if num_barcos < clusters:
    print(f"\n⚠️  ADVERTENCIA: {num_barcos} barcos para {clusters} clusters")
    print(f"   Algunos clusters NO tendrán barco asignado")
elif num_barcos == clusters:
    print(f"\n✅ Configuración balanceada: {num_barcos} barcos = {clusters} clusters")
else:
    print(f"\n✅ {num_barcos} barcos para {clusters} clusters (barcos extra disponibles)")


# FUNCIONES AUXILIARES DE SIMULACIÓN
def vecino_mas_cercano(nodo_actual, nodos_pendientes, nodos_df):
    """Encuentra el nodo más cercano entre los pendientes"""
    if not nodos_pendientes:
        return None, 0
    
    nodo_actual_info = nodos_df[nodos_df['ID'] == nodo_actual].iloc[0]
    x_actual, y_actual = nodo_actual_info['x'], nodo_actual_info['y']
    
    min_distancia = float('inf')
    nodo_mas_cercano = None
    
    for nodo_id in nodos_pendientes:
        nodo_info = nodos_df[nodos_df['ID'] == nodo_id].iloc[0]
        distancia = calcular_distancia(x_actual, y_actual, nodo_info['x'], nodo_info['y'])
        
        if distancia < min_distancia:
            min_distancia = distancia
            nodo_mas_cercano = nodo_id
    
    return nodo_mas_cercano, min_distancia

def inicializar_sistema():
    print("\n" + "="*80)
    print("INICIALIZANDO SISTEMA")
    print("="*80)
    
    # Inicializar inventarios
    inventarios = {}
    for _, consumidor in consumidores.iterrows():
        cons_id = consumidor['ID']
        inventarios[cons_id] = {
            'inventario_actual': 0.0,
            's': consumidores_info[consumidores_info['Consumer'] == cons_id]['s_calculado'].iloc[0],
            'S': consumidores_info[consumidores_info['Consumer'] == cons_id]['S_final'].iloc[0],
            'demanda_semanal': consumidores_info[consumidores_info['Consumer'] == cons_id]['Demanda_Semanal'].iloc[0],
            'capacidad': consumidores_info[consumidores_info['Consumer'] == cons_id]['Capacity'].iloc[0],
            'cluster': consumidor['cluster'],
            'demanda_insatisfecha_acumulada': 0.0,
            'semanas_con_stockout': 0,
        }
    
    print(f"   ✅ {len(inventarios)} consumidores inicializados (inventario = 0)")
    
    # Inicializar barcos con mejor distribución
    barcos_estado = {}
    clusters_disponibles = sorted(nodos['cluster'].unique())
    
    for i in range(num_barcos):
        # Distribuir barcos cíclicamente entre clusters
        cluster_barco = clusters_disponibles[i % len(clusters_disponibles)]
        
        # Buscar proveedor en el cluster asignado
        proveedores_cluster = proveedores[proveedores['cluster'] == cluster_barco]
        
        if len(proveedores_cluster) > 0:
            proveedor_inicial = proveedores_cluster.iloc[0]['ID']
        else:
            print(f"   ⚠️  Cluster {cluster_barco} sin proveedores")
            if len(proveedores) > 0:
                proveedor_inicial = proveedores.iloc[0]['ID']
                cluster_barco = proveedores.iloc[0]['cluster']
                print(f"      Reasignando barco {i} a cluster {cluster_barco}")
            else:
                raise ValueError("No hay proveedores disponibles")
        
        barcos_estado[i] = {
            'id': i,
            'cluster_asignado': cluster_barco,
            'ubicacion_actual': proveedor_inicial,
            'carga_actual': 0.0
        }
    
    print(f"   ✅ {num_barcos} barcos inicializados")
    
    # Mostrar distribución de barcos por cluster
    barcos_por_cluster = {}
    for barco_id, estado in barcos_estado.items():
        cluster = estado['cluster_asignado']
        if cluster not in barcos_por_cluster:
            barcos_por_cluster[cluster] = []
        barcos_por_cluster[cluster].append(barco_id)
    
    print(f"\n   📊 Distribución de barcos:")
    for cluster_id in sorted(clusters_disponibles):
        barcos = barcos_por_cluster.get(cluster_id, [])
        if len(barcos) > 0:
            print(f"      Cluster {cluster_id}: {len(barcos)} barco(s) → {barcos}")
        else:
            print(f"      Cluster {cluster_id}: ⚠️  SIN BARCOS")
    
    return inventarios, barcos_estado

def ejecutar_reabastecimiento(consumidores_a_reabastecer, inventarios, barcos_estado, 
                               proveedores, consumidores, nodos, capacidad_barco, 
                               semana, registro_viajes, verbose=True):
    
    if verbose:
        print(f"\n🚢 EJECUTANDO REABASTECIMIENTO...")
        print(f"   → Consumidores a reabastecer: {len(consumidores_a_reabastecer)}")
    
    if len(consumidores_a_reabastecer) == 0:
        if verbose:
            print(f"   ✅ No se requiere reabastecimiento")
        return 0.0, 0.0, 0
    
    total_distancia_semana = 0
    total_entregas_semana = 0
    viajes_totales = 0
    
    oferta_disponible_proveedores = {}
    for _, proveedor in proveedores.iterrows():
        oferta_disponible_proveedores[proveedor['ID']] = proveedor['Offer']
    
    if verbose:
        print(f"\n   📊 Oferta disponible por proveedor:")
        for prov_id, oferta in oferta_disponible_proveedores.items():
            prov_cluster = proveedores[proveedores['ID'] == prov_id]['cluster'].iloc[0]
            print(f"      Proveedor {prov_id} (Cluster {prov_cluster}): {oferta:.0f} unidades")
    
    # PROCESAR CADA CLUSTER INDEPENDIENTEMENTE
    for cluster_id in sorted(nodos['cluster'].unique()):
        consumidores_cluster = consumidores[consumidores['cluster'] == cluster_id]
        proveedores_cluster = proveedores[proveedores['cluster'] == cluster_id]
        
        if len(consumidores_cluster) == 0 or len(proveedores_cluster) == 0:
            continue
        
        # Filtrar consumidores pendientes de este cluster
        consumidores_pendientes = [
            c['id'] for c in consumidores_a_reabastecer 
            if inventarios[c['id']]['cluster'] == cluster_id
        ]
        
        if not consumidores_pendientes:
            continue
        
        if verbose:
            print(f"\n   {'─'*70}")
            print(f"   📍 CLUSTER {cluster_id}: {len(consumidores_pendientes)} consumidores pendientes")
            print(f"   {'─'*70}")
        
        # Obtener barco asignado a este cluster
        barco_cluster = None
        for barco_id, estado in barcos_estado.items():
            if estado['cluster_asignado'] == cluster_id:
                barco_cluster = barco_id
                break
        
        if barco_cluster is None:
            if verbose:
                print(f"      ⚠️  NO HAY BARCO ASIGNADO - Cluster sin servicio")
            continue
        
        if verbose:
            print(f"      🚢 Barco {barco_cluster} asignado")
        
        proveedores_disponibles = list(proveedores_cluster['ID'])
        ubicacion_actual = barcos_estado[barco_cluster]['ubicacion_actual']
        carga_barco = barcos_estado[barco_cluster].get('carga_actual', 0.0)
        viaje_numero = 1
        
        # Inicializar proveedor_actual
        if carga_barco > 0:
            proveedor_actual = ubicacion_actual if ubicacion_actual in proveedores_disponibles else proveedores_disponibles[0]
            if verbose:
                print(f"      ℹ️  Carga residual: {carga_barco:.2f} unidades")
        else:
            proveedor_actual = None
        
        # Procesar entregas en el cluster
        while consumidores_pendientes:
            # Cargar si está vacío
            if carga_barco == 0:
                if verbose:
                    print(f"\n      🔄 Viaje {viaje_numero}: Cargando barco...")
                
                # Buscar proveedor más cercano con oferta
                proveedor_encontrado = False
                proveedor_mas_cercano = None
                distancia_minima = float('inf')
                
                nodo_actual_info = nodos[nodos['ID'] == ubicacion_actual].iloc[0]
                x_barco, y_barco = nodo_actual_info['x'], nodo_actual_info['y']
                
                for prov_id in proveedores_disponibles:
                    if oferta_disponible_proveedores[prov_id] > 0:
                        prov_info = proveedores_cluster[proveedores_cluster['ID'] == prov_id].iloc[0]
                        distancia = calcular_distancia(x_barco, y_barco, prov_info['x'], prov_info['y'])
                        
                        if distancia < distancia_minima:
                            distancia_minima = distancia
                            proveedor_mas_cercano = prov_id
                            proveedor_encontrado = True
                
                if not proveedor_encontrado:
                    if verbose:
                        print(f"         ⚠️  Sin oferta ({len(consumidores_pendientes)} pendientes)")
                    break
                
                proveedor_actual = proveedor_mas_cercano
                proveedor_info = proveedores_cluster[proveedores_cluster['ID'] == proveedor_actual].iloc[0]
                
                # Mover al proveedor
                if ubicacion_actual != proveedor_actual:
                    distancia_a_proveedor = calcular_distancia(
                        x_barco, y_barco,
                        proveedor_info['x'], proveedor_info['y']
                    )
                    total_distancia_semana += distancia_a_proveedor
                    ubicacion_actual = proveedor_actual
                    if verbose:
                        print(f"         🚢 A proveedor {proveedor_actual}: {distancia_a_proveedor:.2f}")
                
                # Cargar
                cantidad_a_cargar = min(capacidad_barco, oferta_disponible_proveedores[proveedor_actual])
                carga_barco = cantidad_a_cargar
                oferta_disponible_proveedores[proveedor_actual] -= cantidad_a_cargar
                
                if verbose:
                    print(f"         ✅ Cargado: {carga_barco:.2f}, Restante: {oferta_disponible_proveedores[proveedor_actual]:.2f}")
            
            # Ruta de entregas
            entregas_viaje = []
            carga_inicial_viaje = carga_barco
            distancia_viaje = 0
            entregas_en_ruta = 0
            
            if verbose:
                print(f"\n      🚚 Ruta (Carga: {carga_barco:.2f}):")
            
            while carga_barco > 0 and consumidores_pendientes:
                siguiente_consumidor, distancia = vecino_mas_cercano(
                    ubicacion_actual,
                    consumidores_pendientes,
                    nodos
                )
                
                if siguiente_consumidor is None:
                    break
                
                inventario_consumidor = inventarios[siguiente_consumidor]
                cantidad_necesaria = inventario_consumidor['S'] - inventario_consumidor['inventario_actual']
                cantidad_entregada = min(cantidad_necesaria, carga_barco)
                
                # Entregar
                inventarios[siguiente_consumidor]['inventario_actual'] += cantidad_entregada
                carga_barco -= cantidad_entregada
                distancia_viaje += distancia
                total_distancia_semana += distancia
                entregas_en_ruta += 1
                
                entregas_viaje.append({
                    'consumidor': siguiente_consumidor,
                    'cantidad': cantidad_entregada,
                    'inventario_nuevo': inventarios[siguiente_consumidor]['inventario_actual']
                })
                
                if verbose:
                    print(f"         → Puerto {siguiente_consumidor}: +{cantidad_entregada:.2f} " +
                          f"(Inv: {inventario_consumidor['inventario_actual']:.2f}/{inventario_consumidor['S']:.2f}, " +
                          f"dist: {distancia:.2f})")
                
                ubicacion_actual = siguiente_consumidor
                
                # Remover si llegó a S
                if inventarios[siguiente_consumidor]['inventario_actual'] >= inventario_consumidor['S']:
                    consumidores_pendientes.remove(siguiente_consumidor)
                
                total_entregas_semana += cantidad_entregada
            
            # Registrar viaje
            if entregas_viaje:
                if proveedor_actual is None:
                    proveedor_actual = proveedores_disponibles[0]
                
                registro_viajes.append({
                    'semana': semana,
                    'cluster': cluster_id,
                    'barco': barco_cluster,
                    'viaje_numero': viaje_numero,
                    'proveedor_usado': proveedor_actual,
                    'distancia_total': distancia_viaje,
                    'entregas': entregas_viaje,
                    'carga_inicial': carga_inicial_viaje,
                    'carga_final': carga_barco
                })
                viajes_totales += 1
                
                if verbose:
                    print(f"\n      ✅ Viaje {viaje_numero}: {entregas_en_ruta} entregas, dist: {distancia_viaje:.2f}, carga final: {carga_barco:.2f}")
                
                viaje_numero += 1
            
            # Verificar si puede continuar
            if consumidores_pendientes and carga_barco == 0:
                hay_oferta = any(oferta_disponible_proveedores[p] > 0 for p in proveedores_disponibles)
                if not hay_oferta:
                    if verbose:
                        print(f"      ⚠️  Sin más oferta ({len(consumidores_pendientes)} sin abastecer)")
                    break
        
        # Actualizar estado del barco
        barcos_estado[barco_cluster]['ubicacion_actual'] = ubicacion_actual
        barcos_estado[barco_cluster]['carga_actual'] = carga_barco
    
    if verbose:
        print(f"\n   📊 RESUMEN REABASTECIMIENTO:")
        print(f"      Viajes: {viajes_totales}")
        print(f"      Distancia: {total_distancia_semana:.2f}")
        print(f"      Entregas: {total_entregas_semana:.2f}")
    
    return total_distancia_semana, total_entregas_semana, viajes_totales

def simular_semana(semana, inventarios, barcos_estado, proveedores, consumidores, 
                   nodos, capacidad_barco, costo_inventario, registro_viajes,
                   demanda_semanal_total, verbose=True):
    if verbose:
        print(f"\n{'='*80}")
        print(f"📅 SEMANA {semana}")
        print(f"{'='*80}")
    
    # REVISAR POLÍTICAS
    if verbose:
        print(f"\n📋 PASO 1: REVISANDO POLÍTICA (s, S)...")
    
    consumidores_a_reabastecer = []
    
    for cons_id, inv_data in inventarios.items():
        if inv_data['inventario_actual'] <= inv_data['s']:
            cantidad_pedida = inv_data['S'] - inv_data['inventario_actual']
            consumidores_a_reabastecer.append({
                'id': cons_id,
                'inventario_actual': inv_data['inventario_actual'],
                's': inv_data['s'],
                'S': inv_data['S'],
                'cantidad_pedida': cantidad_pedida,
                'prioridad': (inv_data['s'] - inv_data['inventario_actual']) / inv_data['s'] if inv_data['s'] > 0 else 0
            })
    
    consumidores_a_reabastecer.sort(key=lambda x: x['prioridad'], reverse=True)
    
    if verbose:
        print(f"   → Necesitan reabastecimiento: {len(consumidores_a_reabastecer)}/{len(inventarios)}")
        if len(consumidores_a_reabastecer) > 0:
            demanda_reab = sum(c['cantidad_pedida'] for c in consumidores_a_reabastecer)
            print(f"   → Demanda: {demanda_reab:.0f}")
    
    # REABASTECIMIENTO
    if verbose:
        print(f"\n{'─'*80}")
        print(f"PASO 2: REABASTECIMIENTO")
        print(f"{'─'*80}")
    
    total_distancia_semana, total_entregas_semana, viajes_realizados = ejecutar_reabastecimiento(
        consumidores_a_reabastecer=consumidores_a_reabastecer,
        inventarios=inventarios,
        barcos_estado=barcos_estado,
        proveedores=proveedores,
        consumidores=consumidores,
        nodos=nodos,
        capacidad_barco=capacidad_barco,
        semana=semana,
        registro_viajes=registro_viajes,
        verbose=verbose
    )
    
    # CONSUMO
    if verbose:
        print(f"\n{'─'*80}")
        print(f"PASO 3: CONSUMO DE DEMANDA")
        print(f"{'─'*80}")
    
    demanda_no_satisfecha = 0
    puertos_con_faltante = 0
    detalle_stockouts = []  
    stockouts_por_cluster = {} 
    
    for cons_id, inv_data in inventarios.items():
        demanda_semanal = inv_data['demanda_semanal']
        cluster_id = inv_data['cluster']
        
        if inv_data['inventario_actual'] >= demanda_semanal:
            inv_data['inventario_actual'] -= demanda_semanal
        else:
            # STOCKOUT - Demanda parcial o totalmente insatisfecha
            faltante = demanda_semanal - inv_data['inventario_actual']
            demanda_no_satisfecha += faltante
            puertos_con_faltante += 1
            inv_data['demanda_insatisfecha_acumulada'] += faltante
            inv_data['semanas_con_stockout'] += 1
            detalle_stockouts.append({
                'consumidor': cons_id,
                'cluster': cluster_id,
                'demanda_semanal': demanda_semanal,
                'inventario_disponible': inv_data['inventario_actual'],
                'faltante': faltante,
                'porcentaje_faltante': (faltante / demanda_semanal * 100) if demanda_semanal > 0 else 0
            })
            if cluster_id not in stockouts_por_cluster:
                stockouts_por_cluster[cluster_id] = {
                    'num_puertos': 0,
                    'faltante_total': 0
                }
            stockouts_por_cluster[cluster_id]['num_puertos'] += 1
            stockouts_por_cluster[cluster_id]['faltante_total'] += faltante
            
            inv_data['inventario_actual'] = 0 
    
    nivel_servicio = ((demanda_semanal_total - demanda_no_satisfecha)/demanda_semanal_total)*100
    
    if verbose:
        if puertos_con_faltante > 0:
            print(f"   ⚠️  {puertos_con_faltante} puertos con stockout")
            print(f"   📉 Demanda insatisfecha: {demanda_no_satisfecha:.2f} unidades ({(demanda_no_satisfecha/demanda_semanal_total*100):.1f}% del total)")
            if len(detalle_stockouts) > 0:
                detalle_stockouts_sorted = sorted(detalle_stockouts, key=lambda x: x['faltante'], reverse=True)
            if len(stockouts_por_cluster) > 0:
                print(f"\n   📍 Demanda insatisfecha por cluster:")
                for cluster_id in sorted(stockouts_por_cluster.keys()):
                    data = stockouts_por_cluster[cluster_id]
                    print(f"      Cluster {cluster_id}: {data['faltante_total']:.2f} unidades, {data['num_puertos']} puertos afectados")
        else:
            print(f"   ✅ Todos los puertos satisficieron su demanda")
        print(f"   → Nivel de Servicio: {nivel_servicio:.1f}%")
    
    # MÉTRICAS
    if verbose:
        print(f"\n{'─'*80}")
        print(f"PASO 4: MÉTRICAS")
        print(f"{'─'*80}")
    
    inventario_total_puertos = sum(inv['inventario_actual'] for inv in inventarios.values())
    inventario_total_barcos = sum(estado['carga_actual'] for estado in barcos_estado.values())
    inventario_total = inventario_total_puertos + inventario_total_barcos
    
    costo_inventario_puertos = inventario_total_puertos * costo_inventario
    costo_inventario_barcos = inventario_total_barcos * costo_inventario
    costo_inventario_total = costo_inventario_puertos + costo_inventario_barcos
    costo_transporte = total_distancia_semana
    costo_total = costo_inventario_total + costo_transporte
    
    metricas = {
        'semana': semana,
        'demanda_satisfecha': demanda_semanal_total - demanda_no_satisfecha,
        'demanda_no_satisfecha': demanda_no_satisfecha,
        'nivel_servicio': nivel_servicio,
        'puertos_con_stockout': puertos_con_faltante,
        'inventario_puertos': inventario_total_puertos,
        'inventario_barcos': inventario_total_barcos,
        'inventario_total': inventario_total,
        'costo_inventario_puertos': costo_inventario_puertos,
        'costo_inventario_barcos': costo_inventario_barcos,
        'costo_inventario': costo_inventario_total,
        'distancia_recorrida': total_distancia_semana,
        'costo_transporte': costo_transporte,
        'costo_total': costo_total,
        'entregas_realizadas': total_entregas_semana,
        'consumidores_reabastecidos': len(consumidores_a_reabastecer),
        'viajes_realizados': viajes_realizados,
        'detalle_stockouts': detalle_stockouts,
        'stockouts_por_cluster': stockouts_por_cluster,
    }
    
    if verbose:
        print(f"\n   💰 NS={nivel_servicio:.1f}%, Inv={inventario_total:.0f}, Costo=${costo_total:.0f}")
    
    return metricas

def simular_horizonte(num_semanas, verbose=True):
    print("\n" + "="*80)
    print(f"🚀 SIMULACIÓN: {num_semanas + 1} SEMANAS (0 a {num_semanas})")
    print("="*80)
    
    inventarios, barcos_estado = inicializar_sistema()
    registro_viajes = []
    metricas_todas = []
    
    for semana in range(0, num_semanas + 1):
        metricas = simular_semana(
            semana, inventarios, barcos_estado, proveedores, consumidores,
            nodos, capacidad_barco, costo_inventario, registro_viajes,
            demanda_semanal_total, verbose
        )
        metricas_todas.append(metricas)
    
    return metricas_todas, registro_viajes, inventarios


print("\n" + "="*80)
print("⚙️  CONFIGURACIÓN DE SIMULACIÓN")
print("="*80)

print("\nOpciones:")
print("  1. Testing (3 semanas)")
print("  2. Rolling (6 semanas)")
print("  3. Completo (52 semanas)")
print("  4. Personalizado")

opcion = input("\nOpción (1-4): ").strip()

if opcion == "1":
    semanas_a_simular = 2
elif opcion == "2":
    semanas_a_simular = VENTANA_ROLLING - 1
elif opcion == "3":
    semanas_a_simular = HORIZONTE_TOTAL - 1
elif opcion == "4":
    semanas_a_simular = int(input("Semanas (sin contar 0): "))
else:
    print("Opción inválida, usando Testing (3 semanas)")
    semanas_a_simular = 1


print(f"\n✅ Simulando semanas 0 a {semanas_a_simular}")
metricas_todas, registro_viajes_todas, inventarios_finales = simular_horizonte(semanas_a_simular, verbose=True)

print("\n" + "="*80)
print(f"📊 RESUMEN FINAL")
print("="*80)

df_metricas = pd.DataFrame(metricas_todas)

df_semana_0 = df_metricas[df_metricas['semana'] == 0]
df_operacion = df_metricas[df_metricas['semana'] > 0]

print(f"\n📦 SEMANA 0 (Inicialización):")
if len(df_semana_0) > 0:
    print(f"   NS: {df_semana_0['nivel_servicio'].iloc[0]:.1f}%")
    print(f"   Demanda insatisfecha: {df_semana_0['demanda_no_satisfecha'].iloc[0]:.2f} unidades")
    print(f"   Inventario final: {df_semana_0['inventario_total'].iloc[0]:.0f}")
    print(f"   Costo: ${df_semana_0['costo_total'].iloc[0]:.0f}")

if len(df_operacion) > 0:
    print(f"\n📈 OPERACIÓN (Semanas 1-{semanas_a_simular}):")
    print(f"   NS promedio: {df_operacion['nivel_servicio'].mean():.2f}%")
    print(f"   NS mínimo: {df_operacion['nivel_servicio'].min():.2f}% (Semana {df_operacion.loc[df_operacion['nivel_servicio'].idxmin(), 'semana']:.0f})")
    print(f"   NS máximo: {df_operacion['nivel_servicio'].max():.2f}%")

print(f"\n💰 COSTOS TOTALES ({len(df_metricas)} semanas):")
print(f"   Total: ${df_metricas['costo_total'].sum():.2f}")
print(f"   Inventario: ${df_metricas['costo_inventario'].sum():.2f}")
print(f"      - En Puertos: ${df_metricas['costo_inventario_puertos'].sum():.2f}")
print(f"      - En Barcos: ${df_metricas['costo_inventario_barcos'].sum():.2f}")
print(f"   Transporte: ${df_metricas['costo_transporte'].sum():.2f}")
print(f"   Promedio/semana: ${df_metricas['costo_total'].mean():.2f}")

print(f"\n🚢 OPERACIONES:")
print(f"   Viajes totales: {df_metricas['viajes_realizados'].sum():.0f}")
print(f"   Distancia total: {df_metricas['distancia_recorrida'].sum():.2f} unidades")
print(f"   Entregas totales: {df_metricas['entregas_realizadas'].sum():.2f} unidades")


print(f"\n{'='*80}")
print(f"📉 ANÁLISIS DE DEMANDA INSATISFECHA")
print(f"{'='*80}")

total_demanda = demanda_semanal_total * len(df_metricas)
total_demanda_satisfecha = df_metricas['demanda_satisfecha'].sum()
total_demanda_insatisfecha = df_metricas['demanda_no_satisfecha'].sum()

print(f"\n📊 TOTALES:")
print(f"   Demanda total del periodo: {total_demanda:.2f} unidades")
print(f"   Demanda satisfecha: {total_demanda_satisfecha:.2f} unidades ({(total_demanda_satisfecha/total_demanda*100):.2f}%)")
print(f"   Demanda insatisfecha: {total_demanda_insatisfecha:.2f} unidades ({(total_demanda_insatisfecha/total_demanda*100):.2f}%)")

semanas_con_stockout = len(df_metricas[df_metricas['puertos_con_stockout'] > 0])
print(f"\n📅 POR SEMANA:")
print(f"   Semanas con stockout: {semanas_con_stockout}/{len(df_metricas)}")
print(f"   Promedio de demanda insatisfecha/semana: {df_metricas['demanda_no_satisfecha'].mean():.2f} unidades")
if df_metricas['demanda_no_satisfecha'].max() > 0:
    print(f"   Máxima demanda insatisfecha: {df_metricas['demanda_no_satisfecha'].max():.2f} unidades (Semana {df_metricas.loc[df_metricas['demanda_no_satisfecha'].idxmax(), 'semana']:.0f})")

if total_demanda_insatisfecha > 0:
    print(f"\n🔴 TOP 10 PUERTOS CON MÁS DEMANDA INSATISFECHA:")
    puertos_stockout = []
    for cons_id, inv_data in inventarios_finales.items():
        if inv_data['demanda_insatisfecha_acumulada'] > 0:
            puertos_stockout.append({
                'puerto': cons_id,
                'cluster': inv_data['cluster'],
                'demanda_insatisfecha': inv_data['demanda_insatisfecha_acumulada'],
                'semanas_afectadas': inv_data['semanas_con_stockout'],
                'porcentaje_tiempo': (inv_data['semanas_con_stockout'] / len(df_metricas)) * 100,
                'promedio_faltante': inv_data['demanda_insatisfecha_acumulada'] / inv_data['semanas_con_stockout'] if inv_data['semanas_con_stockout'] > 0 else 0
            })
    
    puertos_stockout_sorted = sorted(puertos_stockout, key=lambda x: x['demanda_insatisfecha'], reverse=True)
    
    for i, puerto in enumerate(puertos_stockout_sorted[:10], 1):
        print(f"   {i}. Puerto {puerto['puerto']} (Cluster {puerto['cluster']}):")
        print(f"      • Demanda insatisfecha acumulada: {puerto['demanda_insatisfecha']:.2f} unidades")
        print(f"      • Semanas afectadas: {puerto['semanas_afectadas']}/{len(df_metricas)} ({puerto['porcentaje_tiempo']:.1f}% del tiempo)")
        print(f"      • Promedio por semana con stockout: {puerto['promedio_faltante']:.2f} unidades")
    
    print(f"\n📍 DEMANDA INSATISFECHA POR CLUSTER:")
    stockouts_cluster_total = {}
    for metricas in metricas_todas:
        for cluster_id, data in metricas['stockouts_por_cluster'].items():
            if cluster_id not in stockouts_cluster_total:
                stockouts_cluster_total[cluster_id] = {
                    'puertos_semana': 0,
                    'faltante_total': 0,
                    'semanas_con_stockout': 0
                }
            stockouts_cluster_total[cluster_id]['puertos_semana'] += data['num_puertos']
            stockouts_cluster_total[cluster_id]['faltante_total'] += data['faltante_total']
            stockouts_cluster_total[cluster_id]['semanas_con_stockout'] += 1
    
    for cluster_id in sorted(stockouts_cluster_total.keys()):
        data = stockouts_cluster_total[cluster_id]
        print(f"\n   Cluster {cluster_id}:")
        print(f"      • Demanda insatisfecha total: {data['faltante_total']:.2f} unidades ({(data['faltante_total']/total_demanda_insatisfecha*100):.1f}% del total)")
        print(f"      • Semanas con stockout: {data['semanas_con_stockout']}/{len(df_metricas)}")
        print(f"      • Puertos-semana afectados: {data['puertos_semana']}")
        print(f"      • Promedio/semana con stockout: {data['faltante_total']/data['semanas_con_stockout']:.2f} unidades")
    
    clusters_sin_stockout = set(consumidores['cluster'].unique()) - set(stockouts_cluster_total.keys())
    if clusters_sin_stockout:
        print(f"\n   ✅ Clusters SIN demanda insatisfecha: {sorted(clusters_sin_stockout)}")
else:
    print(f"\n✅ ¡PERFECTO! No hubo demanda insatisfecha en ninguna semana")
    print(f"   Todos los puertos fueron abastecidos correctamente")

print("\n" + "="*80)
print("✅ SIMULACIÓN COMPLETADA")
print("="*80)