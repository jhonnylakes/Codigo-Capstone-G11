
import pandas as pd
import numpy as np
import random
import copy
import re


def cargar_y_procesar_datos():
    """
    Lee tus archivos, genera IDs únicos para evitar errores y devuelve los datos listos.
    """
    try:
        additional_df = pd.read_csv('Datos/additional_data.csv', header=None, skiprows=1)
        params = {
            'CAPACIDAD_BARCO': int(additional_df.iloc[0, 1]),
            'COSTO_INVENTARIO': float(additional_df.iloc[0, 2]),
            'N_BARCOS': int(additional_df.iloc[0, 3])
        }
        nodos_df = pd.read_csv('Datos/nodos_con_clusters_optimizado.csv')
        producer_df = nodos_df[nodos_df['Tipo'] == 'Proveedor'].copy().reset_index(drop=True)
        consumer_df = nodos_df[nodos_df['Tipo'] == 'Consumidor'].copy().reset_index(drop=True)
        producer_df['id'] = ['P' + str(i) for i in producer_df.index]
        consumer_df['id'] = ['C' + str(i) for i in consumer_df.index]
        producer_df = producer_df.rename(columns={'Offer': 'oferta'})
        consumer_df = consumer_df.rename(columns={'Demand': 'demanda', 'Capacity': 'capacidad'})
        cluster_to_producer_map = producer_df.set_index('cluster')['id'].to_dict()
        consumer_df['productor_asignado'] = consumer_df['cluster'].map(cluster_to_producer_map)
        puertos_df = pd.concat([
            producer_df[['id', 'x', 'y']],
            consumer_df[['id', 'x', 'y']]
        ]).set_index('id')
        print("✓ Datos y clusters cargados correctamente.")
        return producer_df, consumer_df, puertos_df, params
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo '{e.filename}'. Asegúrate de que la carpeta 'Datos/' exista y contenga los archivos.")
        return None, None, None, None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer los datos: {e}")
        return None, None, None, None

# Parámetros 
COSTO_VIAJE = 4.0
INGRESO_ENTREGA = 5.0
PENALIZACION_DNS = 1.0
TAMANO_POBLACION = 50
PROBABILIDAD_MUTACION = 0.2
N_GENERACIONES = 20
TS_ITERACIONES = 10
TS_TAMANO_LISTA = 7

#  (HGA-TS) 

def calcular_distancia(p1_id, p2_id, puertos):
    if p1_id is None or p2_id is None: return 0
    try:
        p1 = puertos.loc[p1_id]
        p2 = puertos.loc[p2_id]
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    except KeyError: 
        return 0

def encontrar_puerto_mas_cercano(puerto_actual_id, lista_puertos, puertos_df):
    if not lista_puertos or puerto_actual_id is None: return None, float('inf')
    mejor_puerto, menor_distancia = None, float('inf')
    for puerto in lista_puertos:
        dist = calcular_distancia(puerto_actual_id, puerto['id'], puertos_df)
        if dist < menor_distancia:
            menor_distancia, mejor_puerto = dist, puerto
    return mejor_puerto, menor_distancia

def crear_individuo_economico(productores_df, consumidores_df, n_barcos, capacidad, puertos_df, estado_inventarios):
    """
    Crea un individuo que solo atiende la demanda pendiente y considera costos.
    """
    rutas = []

    consumidores_pendientes = []
    for _, cons in consumidores_df.iterrows():
        demanda_pendiente = cons['demanda'] - estado_inventarios.get(cons['id'], 0)
        if demanda_pendiente > 0:
            cons_copy = cons.to_dict()
            cons_copy['demanda_pendiente'] = demanda_pendiente
            consumidores_pendientes.append(cons_copy)

    for i in range(1, int(n_barcos) + 1):
        ruta_buque = {'buque_id': f'B{i}', 'ruta': []}
        productor_inicial = productores_df.sample(1).iloc[0]
        ubicacion_actual = productor_inicial['id']
        carga_actual = 0
        max_paradas = 24

        while len(ruta_buque['ruta']) < max_paradas and consumidores_pendientes:
            min_demanda_pendiente = min(c['demanda_pendiente'] for c in consumidores_pendientes) if consumidores_pendientes else 0
            
            if carga_actual < (capacidad * 0.25) or (min_demanda_pendiente > 0 and carga_actual < min_demanda_pendiente):
                productor_cercano, _ = encontrar_puerto_mas_cercano(ubicacion_actual, productores_df.to_dict('records'), puertos_df)
                if productor_cercano is None: break
                
                cantidad_a_cargar = capacidad - carga_actual
                ruta_buque['ruta'].append({'puerto_id': productor_cercano['id'], 'tipo': 'carga', 'cantidad': cantidad_a_cargar})
                carga_actual += cantidad_a_cargar
                ubicacion_actual = productor_cercano['id']

            if not consumidores_pendientes: break

            consumidor_cercano, dist_a_consumidor = encontrar_puerto_mas_cercano(ubicacion_actual, consumidores_pendientes, puertos_df)
            if consumidor_cercano is None: break
            
            demanda_a_entregar = consumidor_cercano['demanda_pendiente']

            if carga_actual >= demanda_a_entregar:
                costo_del_tramo = dist_a_consumidor * COSTO_VIAJE
                ingreso_del_tramo = demanda_a_entregar * INGRESO_ENTREGA
                
                if costo_del_tramo > ingreso_del_tramo and len(ruta_buque['ruta']) > 1:
                    break 

                ruta_buque['ruta'].append({'puerto_id': consumidor_cercano['id'], 'tipo': 'descarga', 'cantidad': demanda_a_entregar})
                carga_actual -= demanda_a_entregar
                ubicacion_actual = consumidor_cercano['id']
                consumidores_pendientes.remove(consumidor_cercano)
            else:
                carga_actual = 0 
        
        rutas.append(ruta_buque)
        
    return {'fitness': 0, 'rutas': rutas}

def evaluar_fitness(individuo, puertos, consumidores, estado_inventarios_inicial, params):
    costo_viaje_total = 0
    entregas = {c['id']: 0 for c in consumidores}
    
    for i, plan in enumerate(individuo['rutas']):
        ruta = plan['ruta']
        
        carga_actual = 0 
        for parada in ruta:
            if parada['tipo'] == 'carga':
                carga_actual += parada['cantidad']
                if carga_actual > params['CAPACIDAD_BARCO'] + 1:
                    individuo['fitness'] = -float('inf'); return -float('inf')
            elif parada['tipo'] == 'descarga':
                if carga_actual < parada['cantidad']:
                    individuo['fitness'] = -float('inf'); return -float('inf')
                carga_actual -= parada['cantidad']

        if len(ruta) < 2: continue
        for j in range(len(ruta) - 1):
            costo_viaje_total += calcular_distancia(ruta[j]['puerto_id'], ruta[j+1]['puerto_id'], puertos)
        
        for parada in ruta:
            if parada['tipo'] == 'descarga':
                entregas[parada['puerto_id']] += parada['cantidad']

    ingreso_total = 0
    dns_total = 0
    inventario_final = copy.deepcopy(estado_inventarios_inicial)

    for cons in consumidores:
        cons_id = cons['id']
        demanda_semanal = cons['demanda']
        inv_inicial = estado_inventarios_inicial.get(cons_id, 0)
        entregado = entregas.get(cons_id, 0)

        demanda_pendiente = max(0, demanda_semanal - inv_inicial)
        unidades_utiles_entregadas = min(entregado, demanda_pendiente)
        ingreso_total += unidades_utiles_entregadas * INGRESO_ENTREGA
        dns_total += max(0, demanda_pendiente - entregado)

        inv_final_antes_consumo = inv_inicial + entregado
        inventario_final[cons_id] = max(0, inv_final_antes_consumo - demanda_semanal)

    costo_inv = sum(inv * params['COSTO_INVENTARIO'] for inv in inventario_final.values())
    costo_total = (costo_viaje_total * COSTO_VIAJE) + (dns_total * PENALIZACION_DNS) + costo_inv
    
    individuo['fitness'] = ingreso_total - costo_total
    return individuo['fitness']

def seleccion_por_ruleta(poblacion):
    poblacion_valida = [ind for ind in poblacion if ind.get('fitness', -float('inf')) > -float('inf')]
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
        rutas_elegibles = [r for r in individuo['rutas'] if len(r['ruta']) > 1]
        if not rutas_elegibles: return individuo
        ruta_a_mutar = random.choice(rutas_elegibles)
        paradas_intercambiables = list(range(len(ruta_a_mutar['ruta'])))
        if len(paradas_intercambiables) < 2: return individuo
        idx1, idx2 = random.sample(paradas_intercambiables, 2)
        if ruta_a_mutar['ruta'][idx1]['tipo'] == ruta_a_mutar['ruta'][idx2]['tipo']:
            ruta_a_mutar['ruta'][idx1], ruta_a_mutar['ruta'][idx2] = ruta_a_mutar['ruta'][idx2], ruta_a_mutar['ruta'][idx1]
    return individuo

def generar_vecino_swap(solucion):
    vecino = copy.deepcopy(solucion)
    rutas_elegibles = [i for i, r in enumerate(vecino['rutas']) if len(r['ruta']) > 1]
    if not rutas_elegibles: return None, None
    idx_ruta = random.choice(rutas_elegibles)
    paradas = vecino['rutas'][idx_ruta]['ruta']
    if len(paradas) < 2: return None, None
    idx1, idx2 = random.sample(range(len(paradas)), 2)
    if paradas[idx1]['tipo'] != paradas[idx2]['tipo']: return None, None
    movimiento = ('swap', idx_ruta, paradas[idx1]['puerto_id'], paradas[idx2]['puerto_id'])
    paradas[idx1], paradas[idx2] = paradas[idx2], paradas[idx1]
    return vecino, movimiento

def busqueda_tabu(individuo_inicial, puertos, consumidores, estado_inventarios, params):
    mejor_solucion = copy.deepcopy(individuo_inicial)
    mejor_fitness_global = evaluar_fitness(mejor_solucion, puertos, consumidores, estado_inventarios, params)
    if mejor_fitness_global == -float('inf'): return individuo_inicial
    solucion_actual = copy.deepcopy(mejor_solucion)
    lista_tabu = []
    for _ in range(TS_ITERACIONES):
        mejor_vecino, mejor_fitness_vecino, mejor_movimiento = None, -float('inf'), None
        for _ in range(20):
            vecino, movimiento = generar_vecino_swap(solucion_actual)
            if not vecino or not movimiento: continue
            fitness_vecino = evaluar_fitness(vecino, puertos, consumidores, estado_inventarios, params)
            if fitness_vecino == -float('inf'): continue
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

if __name__ == "__main__":
    
    producer_df, consumer_df, puertos_df, params = cargar_y_procesar_datos()

    if producer_df is not None:
        consumers_list = consumer_df.to_dict('records')
        
        estado_barcos_inicial_dummy = {f'B{i+1}': {'ubicacion': None, 'carga_a_bordo': 0} for i in range(int(params['N_BARCOS']))}
        estado_inventarios_inicial = {row['id']: 0 for _, row in consumer_df.iterrows()}

        print("\n--- INICIANDO ALGORITMO HÍBRIDO (MODO STANDALONE) ---")
        print(f"Parámetros: {params['N_BARCOS']} barcos de {params['CAPACIDAD_BARCO']} de capacidad.")
        print(f"Población: {TAMANO_POBLACION}, Generaciones: {N_GENERACIONES}\n")

        poblacion = [crear_individuo_economico(producer_df, consumer_df, params['N_BARCOS'], params['CAPACIDAD_BARCO'], puertos_df, estado_inventarios_inicial) for _ in range(TAMANO_POBLACION)]
        
        for ind in poblacion:
            evaluar_fitness(ind, puertos_df, consumers_list, estado_inventarios_inicial, params)

        mejor_fitness_global = -float('inf')

        for gen in range(N_GENERACIONES):
            padres = seleccion_por_ruleta(poblacion)
            
            nueva_poblacion = []
            if len(padres) > 1:
                for i in range(0, TAMANO_POBLACION, 2):
                    hijo1, hijo2 = cruzamiento(padres[i], padres[i+1], params['N_BARCOS'])
                    nueva_poblacion.extend([mutacion(hijo1), mutacion(hijo2)])
            else:
                nueva_poblacion = [copy.deepcopy(p) for p in padres]
            
            for ind in nueva_poblacion:
                evaluar_fitness(ind, puertos_df, consumers_list, estado_inventarios_inicial, params)
                
            mejor_de_generacion = max(poblacion, key=lambda x: x.get('fitness', -float('inf')))
            
            mejor_refinado = busqueda_tabu(mejor_de_generacion, puertos_df, consumers_list, estado_inventarios_inicial, params)
            
            if nueva_poblacion:
                peor_nuevo_idx = min(range(len(nueva_poblacion)), key=lambda i: nueva_poblacion[i].get('fitness', -float('inf')))
                if mejor_refinado.get('fitness', -float('inf')) > nueva_poblacion[peor_nuevo_idx].get('fitness', -float('inf')):
                    nueva_poblacion[peor_nuevo_idx] = mejor_refinado
            
            poblacion = nueva_poblacion if nueva_poblacion else poblacion
            
            mejor_fitness_actual = max(ind.get('fitness', -float('inf')) for ind in poblacion)
            if mejor_fitness_actual > mejor_fitness_global:
                mejor_fitness_global = mejor_fitness_actual

            fitness_promedio = np.mean([ind['fitness'] for ind in poblacion if ind.get('fitness', -float('inf')) > -float('inf')])
            
            print(f"Generación {gen+1:2d}: Mejor Fitness = {mejor_fitness_actual:10.2f} (Mejor Global: {mejor_fitness_global:10.2f})")

        print("\n--- EVOLUCIÓN COMPLETADA ---")
        mejor_individuo_final = max(poblacion, key=lambda x: x.get('fitness', -float('inf')))
        print(f"Mejor fitness global encontrado: {mejor_individuo_final['fitness']:.2f}")

        print("\n" + "="*50)
        print("--- ANÁLISIS DEL MEJOR PLAN SEMANAL ENCONTRADO ---")
        print("="*50)

        total_entregado = 0
        consumidores_atendidos = set()
        distancia_total = 0

        for plan_buque in mejor_individuo_final['rutas']:
            ruta = plan_buque['ruta']
            if not ruta: continue
            
            for i in range(len(ruta) - 1):
                distancia_total += calcular_distancia(ruta[i]['puerto_id'], ruta[i+1]['puerto_id'], puertos_df)
            
            for parada in ruta:
                if parada['tipo'] == 'descarga':
                    total_entregado += parada['cantidad']
                    consumidores_atendidos.add(parada['puerto_id'])
        
        print(f"  - Fitness (Utilidad): {mejor_individuo_final['fitness']:.2f}")
        print(f"  - Unidades Totales Entregadas: {int(total_entregado)}")
        print(f"  - Consumidores Únicos Atendidos: {len(consumidores_atendidos)} de {len(consumer_df)}")
        print(f"  - Distancia Total Recorrida: {distancia_total:.2f}")
        # imprimir ruta de los buques y cantidad de paradas
        for plan_buque in mejor_individuo_final['rutas']:
            ruta = plan_buque['ruta']
            buque_id = plan_buque['buque_id']
            if len(ruta) <= 1:
                print(f"\n- Buque {buque_id}: RUTA VACÍA O SOLO CARGA (no realiza entregas)")
                continue
            ruta_str = f"Ruta {buque_id}: "
            for parada in ruta:
                if parada['tipo'] == 'carga':
                    ruta_str += f"{parada['puerto_id']}(C:{parada['cantidad']}) -> "
                else:
                    ruta_str += f"{parada['puerto_id']}(D:{parada['cantidad']}) -> "
            ruta_str = ruta_str.rstrip(" -> ")
            print(f"\n- {ruta_str}")

        
        print("="*50)