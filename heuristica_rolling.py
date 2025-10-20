import pandas as pd
import numpy as np
import random
import copy
import re

# --- 1. CARGA DE DATOS INTELIGENTE DESDE ARCHIVOS (VERSIÓN FINAL Y ROBUSTA) ---

def cargar_y_procesar_datos():
    """
    Lee tus archivos, genera IDs únicos para evitar errores y devuelve los datos listos.
    """
    try:
        # Cargar Parámetros Adicionales
        additional_df = pd.read_csv('Datos/additional_data.csv', header=None, skiprows=1)
        params = {
            'CAPACIDAD_BARCO': int(additional_df.iloc[0, 1]),
            'COSTO_INVENTARIO': float(additional_df.iloc[0, 2]),
            'N_BARCOS': int(additional_df.iloc[0, 3])
        }

        # Cargar tu archivo de nodos pre-clusterizado
        nodos_df = pd.read_csv('Datos/nodos_con_clusters_optimizado.csv')

        # Separar en Productores y Consumidores
        producer_df = nodos_df[nodos_df['Tipo'] == 'Proveedor'].copy().reset_index(drop=True)
        consumer_df = nodos_df[nodos_df['Tipo'] == 'Consumidor'].copy().reset_index(drop=True)

        # --- SOLUCIÓN AL ERROR: Generar IDs únicos garantizados ---
        producer_df['id'] = ['P' + str(i) for i in producer_df.index]
        consumer_df['id'] = ['C' + str(i) for i in consumer_df.index]
        
        # Renombrar columnas para consistencia
        producer_df = producer_df.rename(columns={'Offer': 'oferta'})
        consumer_df = consumer_df.rename(columns={'Demand': 'demanda', 'Capacity': 'capacidad'})
        
        # Mapear el clúster al ID del productor
        cluster_to_producer_map = producer_df.set_index('cluster')['id'].to_dict()
        consumer_df['productor_asignado'] = consumer_df['cluster'].map(cluster_to_producer_map)
        
        # Crear DataFrame unificado para distancias
        puertos_df = pd.concat([
            producer_df[['id', 'x', 'y']],
            consumer_df[['id', 'x', 'y']]
        ]).set_index('id')

        print("✓ Datos y clusters cargados correctamente. IDs únicos generados.")
        return producer_df, consumer_df, puertos_df, params

    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo '{e.filename}'.")
        return None, None, None, None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer los datos: {e}")
        return None, None, None, None

# (El resto del código es idéntico al anterior)
# Parámetros del Algoritmo
COSTO_VIAJE = 1.0
INGRESO_ENTREGA = 1.0
PENALIZACION_DNS = 2.0
TAMANO_POBLACION = 50
PROBABILIDAD_MUTACION = 0.2
N_GENERACIONES = 20
TS_ITERACIONES = 10
TS_TAMANO_LISTA = 7

# --- AQUÍ VAN TODAS LAS FUNCIONES QUE YA CONSTRUIMOS ---
# (calcular_distancia, evaluar_fitness, seleccion_por_ruleta, cruzamiento, mutacion, busqueda_tabu, etc.)

def calcular_distancia(p1_id, p2_id, puertos):
    try:
        p1 = puertos.loc[p1_id]
        p2 = puertos.loc[p2_id]
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    except KeyError: 
        return 0

def crear_individuo_inteligente(productores, consumidores_df, n_barcos, capacidad, estado_barcos):
    rutas = []
    consumidores_libres = consumidores_df.to_dict('records')
    min_demanda_global = consumidores_df['demanda'].min()

    for i in range(1, int(n_barcos) + 1):
        buque_id = f'B{i}'
        ruta_buque = {'buque_id': buque_id, 'ruta': []}
        
        # MODIFICACIÓN CLAVE: El barco ya no empieza en un productor aleatorio,
        # sino que su primera "parada" es su ubicación actual.
        ubicacion_actual = estado_barcos[buque_id]['ubicacion']
        
        if not productores.empty:
            # Decide a qué productor ir desde su ubicación actual (podríamos usar lógica más avanzada)
            productor = productores.sample(1).iloc[0]
            ruta_buque['ruta'].append({'puerto_id': productor['id'], 'tipo': 'carga', 'cantidad': capacidad})
            carga = capacidad

            # El resto de la lógica para asignar consumidores del clúster sigue igual
            consumidores_cluster = [c for c in consumidores_libres if c.get('productor_asignado') == productor['id']]
            random.shuffle(consumidores_cluster)
            
            visitados = []
            for consumidor in consumidores_cluster:
                if carga >= consumidor['demanda']:
                    ruta_buque['ruta'].append({'puerto_id': consumidor['id'], 'tipo': 'descarga', 'cantidad': consumidor['demanda']})
                    carga -= consumidor['demanda']
                    visitados.append(consumidor)
                if carga < min_demanda_global: break
            
            consumidores_libres = [c for c in consumidores_libres if c not in visitados]
        rutas.append(ruta_buque)
    return {'fitness': 0, 'rutas': rutas}

def evaluar_fitness(individuo, puertos, consumidores, estado_barcos):
    costo_viaje_total, ingreso_total = 0, 0
    entregas = {c['id']: 0 for c in consumidores}
    for i, plan in enumerate(individuo['rutas']):
        ruta = plan['ruta']
        buque_id = f'B{i+1}'
        
        # Validar ruta y calcular costos
        carga_actual = 0
        if not ruta: continue

        # MODIFICACIÓN: El costo del primer tramo es desde la ubicación inicial del barco
        ubicacion_inicial = estado_barcos[buque_id]['ubicacion']
        costo_viaje_total += calcular_distancia(ubicacion_inicial, ruta[0]['puerto_id'], puertos)

        for j in range(len(ruta) - 1):
            costo_viaje_total += calcular_distancia(ruta[j]['puerto_id'], ruta[j+1]['puerto_id'], puertos)
        
        for parada in ruta:
            if parada['tipo'] == 'carga': carga_actual += parada['cantidad']
            elif parada['tipo'] == 'descarga':
                if carga_actual < parada['cantidad']:
                    individuo['fitness'] = -float('inf'); return -float('inf')
                carga_actual -= parada['cantidad']
                ingreso_total += parada['cantidad'] * INGRESO_ENTREGA
                if parada['puerto_id'] in entregas: entregas[parada['puerto_id']] += parada['cantidad']
    
    dns = sum(max(0, c['demanda'] - entregas.get(c['id'], 0)) for c in consumidores)
    costo_total = (costo_viaje_total * COSTO_VIAJE) + (dns * PENALIZACION_DNS)
    individuo['fitness'] = ingreso_total - costo_total
    return individuo['fitness']

# (Aquí irían el resto de funciones: seleccion, cruzamiento, mutacion, busqueda_tabu, etc., sin cambios)
def seleccion_por_ruleta(poblacion):
    poblacion_valida = [ind for ind in poblacion if ind['fitness'] > -float('inf')]
    if not poblacion_valida: return [copy.deepcopy(random.choice(poblacion)) for _ in range(len(poblacion))]
    min_fitness = min(ind['fitness'] for ind in poblacion_valida)
    desplazamiento = abs(min_fitness) + 1 if min_fitness <= 0 else 0
    total_fitness_ajustado = sum(ind['fitness'] + desplazamiento for ind in poblacion_valida)
    if total_fitness_ajustado == 0: return [copy.deepcopy(random.choice(poblacion_valida)) for _ in range(len(poblacion))]
    seleccionados = []
    for _ in range(len(poblacion)):
        pick = random.uniform(0, total_fitness_ajustado)
        current = 0
        for individuo in poblacion_valida:
            current += individuo['fitness'] + desplazamiento
            if current > pick:
                seleccionados.append(copy.deepcopy(individuo))
                break
    return seleccionados

def cruzamiento(padre1, padre2, n_barcos):
    hijo1, hijo2 = copy.deepcopy(padre1), copy.deepcopy(padre2)
    punto_corte = random.randint(1, int(n_barcos) - 1)
    hijo1['rutas'][punto_corte:] = padre2['rutas'][punto_corte:]
    hijo2['rutas'][punto_corte:] = padre1['rutas'][punto_corte:]
    return hijo1, hijo2

def mutacion(individuo):
    if random.random() < PROBABILIDAD_MUTACION:
        rutas_elegibles = [r for r in individuo['rutas'] if len(r['ruta']) > 2]
        if not rutas_elegibles: return individuo
        ruta_a_mutar = random.choice(rutas_elegibles)
        descargas = ruta_a_mutar['ruta'][1:]
        if len(descargas) < 2: return individuo
        idx1, idx2 = random.sample(range(len(descargas)), 2)
        descargas[idx1], descargas[idx2] = descargas[idx2], descargas[idx1]
        ruta_a_mutar['ruta'][1:] = descargas
    return individuo

def generar_vecino_swap(solucion):
    vecino = copy.deepcopy(solucion)
    rutas_con_descargas = [i for i, r in enumerate(vecino['rutas']) if len(r['ruta']) > 2]
    if not rutas_con_descargas: return None, None
    idx_ruta = random.choice(rutas_con_descargas)
    descargas = vecino['rutas'][idx_ruta]['ruta'][1:]
    if len(descargas) < 2: return None, None
    idx1, idx2 = random.sample(range(len(descargas)), 2)
    movimiento = ('swap', idx_ruta, descargas[idx1]['puerto_id'], descargas[idx2]['puerto_id'])
    descargas[idx1], descargas[idx2] = descargas[idx2], descargas[idx1]
    vecino['rutas'][idx_ruta]['ruta'][1:] = descargas
    return vecino, movimiento

def busqueda_tabu(individuo_inicial, puertos, consumidores, estado_barcos):
    mejor_solucion = copy.deepcopy(individuo_inicial)
    mejor_fitness_global = evaluar_fitness(mejor_solucion, puertos, consumidores, estado_barcos)
    if mejor_fitness_global == -float('inf'): return individuo_inicial
    solucion_actual = copy.deepcopy(mejor_solucion)
    lista_tabu = []
    for _ in range(TS_ITERACIONES):
        mejor_vecino, mejor_fitness_vecino, mejor_movimiento = None, -float('inf'), None
        for _ in range(20):
            vecino, movimiento = generar_vecino_swap(solucion_actual)
            if not vecino or not movimiento: continue
            fitness_vecino = evaluar_fitness(vecino, puertos, consumidores, estado_barcos)
            es_mejor_global = fitness_vecino > mejor_fitness_global
            if (movimiento not in lista_tabu) or es_mejor_global:
                if fitness_vecino > mejor_fitness_vecino:
                    mejor_fitness_vecino, mejor_vecino, mejor_movimiento = fitness_vecino, vecino, movimiento
        if mejor_vecino:
            solucion_actual = mejor_vecino
            lista_tabu.append(mejor_movimiento)
            if len(lista_tabu) > TS_TAMANO_LISTA: lista_tabu.pop(0)
            if mejor_fitness_vecino > mejor_fitness_global:
                mejor_solucion, mejor_fitness_global = mejor_vecino, mejor_fitness_vecino
    return mejor_solucion

# --- FUNCIÓN PRINCIPAL DEL MÓDULO ---
def ejecutar_optimizacion_semanal(producer_df, consumer_df, puertos_df, params, estado_barcos):
    """
    Ejecuta el HGA-TS para una semana y devuelve el mejor plan encontrado.
    """
    consumers_list = consumer_df.to_dict('records')
    
    poblacion = [crear_individuo_inteligente(producer_df, consumer_df, params['N_BARCOS'], params['CAPACIDAD_BARCO'], estado_barcos) for _ in range(TAMANO_POBLACION)]
    for ind in poblacion:
        evaluar_fitness(ind, puertos_df, consumers_list, estado_barcos)

    mejor_fitness_global = -float('inf')
    mejor_individuo_global = None

    for gen in range(N_GENERACIONES):
        padres = seleccion_por_ruleta(poblacion)
        
        nueva_poblacion = []
        for i in range(0, TAMANO_POBLACION, 2):
            hijo1, hijo2 = cruzamiento(padres[i], padres[i+1], params['N_BARCOS'])
            nueva_poblacion.extend([mutacion(hijo1), mutacion(hijo2)])
        
        for ind in nueva_poblacion:
            evaluar_fitness(ind, puertos_df, consumers_list, estado_barcos)
            
        mejor_de_generacion = max(poblacion, key=lambda x: x['fitness'])
        mejor_refinado = busqueda_tabu(mejor_de_generacion, puertos_df, consumers_list, estado_barcos)
        
        peor_nuevo_idx = min(range(len(nueva_poblacion)), key=lambda i: nueva_poblacion[i]['fitness'])
        if mejor_refinado['fitness'] > nueva_poblacion[peor_nuevo_idx]['fitness']:
            nueva_poblacion[peor_nuevo_idx] = mejor_refinado
        
        poblacion = nueva_poblacion
        
        mejor_fitness_actual = max(ind['fitness'] for ind in poblacion)
        if mejor_fitness_actual > mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_actual
            mejor_individuo_global = max(poblacion, key=lambda x: x['fitness'])
    
    return mejor_individuo_global