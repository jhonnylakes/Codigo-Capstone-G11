import pandas as pd
import numpy as np
import random
import copy
import re
import matplotlib.pyplot as plt


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

        print("✓ Datos y clusters cargados correctamente. IDs únicos generados.")
        return producer_df, consumer_df, puertos_df, params

    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo '{e.filename}'.")
        return None, None, None, None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer los datos: {e}")
        return None, None, None, None



def calcular_distancia(p1_id, p2_id, puertos):
    try:
        p1 = puertos.loc[p1_id]
        p2 = puertos.loc[p2_id]
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    except KeyError: 
        return 0

def crear_individuo_inteligente(productores, consumidores_df, n_barcos, capacidad):
    rutas = []
    consumidores_libres = consumidores_df.to_dict('records')
    min_demanda_global = consumidores_df['demanda'].min()

    for i in range(1, int(n_barcos) + 1):
        ruta_buque = {'buque_id': f'B{i}', 'ruta': []}
        if not productores.empty:
            productor = productores.sample(1).iloc[0]
            ruta_buque['ruta'].append({'puerto_id': productor['id'], 'tipo': 'carga', 'cantidad': capacidad})
            carga = capacidad

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


def evaluar_fitness(individuo, puertos, consumidores, costo_viaje, ingreso_entrega, penalizacion_dns):
    costo_viaje_total, ingreso_total = 0, 0
    entregas = {c['id']: 0 for c in consumidores}
    for plan in individuo['rutas']:
        ruta = plan['ruta']
        carga_actual = 0
        ruta_valida = True
        for parada in ruta:
            if parada['tipo'] == 'carga':
                carga_actual += parada['cantidad']
            elif parada['tipo'] == 'descarga':
                if carga_actual < parada['cantidad']:
                    ruta_valida = False
                    break
                carga_actual -= parada['cantidad']
        if not ruta_valida:
            individuo['fitness'] = -float('inf')
            return -float('inf')

        if len(ruta) < 2: continue
        for i in range(len(ruta) - 1):
            costo_viaje_total += calcular_distancia(ruta[i]['puerto_id'], ruta[i+1]['puerto_id'], puertos)
        for parada in ruta:
            if parada['tipo'] == 'descarga':
                ingreso_total += parada['cantidad'] * ingreso_entrega
                if parada['puerto_id'] in entregas: entregas[parada['puerto_id']] += parada['cantidad']
    
    dns = sum(max(0, c['demanda'] - entregas.get(c['id'], 0)) for c in consumidores)
    
    costo_total = (costo_viaje_total * costo_viaje) + (dns * penalizacion_dns)
    individuo['fitness'] = ingreso_total - costo_total
    return individuo['fitness']

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

def mutacion(individuo, probabilidad_mutacion):
    if random.random() < probabilidad_mutacion:
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


def busqueda_tabu(individuo_inicial, puertos, consumidores, costo_viaje, ingreso_entrega, penalizacion_dns, ts_iteraciones, ts_tamano_lista):
    mejor_solucion = copy.deepcopy(individuo_inicial)
    mejor_fitness_global = evaluar_fitness(mejor_solucion, puertos, consumidores, costo_viaje, ingreso_entrega, penalizacion_dns)
    if mejor_fitness_global == -float('inf'): return individuo_inicial
    
    solucion_actual = copy.deepcopy(mejor_solucion)
    lista_tabu = []
    
    for _ in range(ts_iteraciones):
        mejor_vecino, mejor_fitness_vecino, mejor_movimiento = None, -float('inf'), None
        for _ in range(20): 
            vecino, movimiento = generar_vecino_swap(solucion_actual)
            if not vecino or not movimiento: continue
            
            fitness_vecino = evaluar_fitness(vecino, puertos, consumidores, costo_viaje, ingreso_entrega, penalizacion_dns)
            
            es_mejor_global = fitness_vecino > mejor_fitness_global
            if (movimiento not in lista_tabu) or es_mejor_global:
                if fitness_vecino > mejor_fitness_vecino:
                    mejor_fitness_vecino, mejor_vecino, mejor_movimiento = fitness_vecino, vecino, movimiento
        
        if mejor_vecino:
            solucion_actual = mejor_vecino
            lista_tabu.append(mejor_movimiento)
            if len(lista_tabu) > ts_tamano_lista: lista_tabu.pop(0)
            
            if mejor_fitness_vecino > mejor_fitness_global:
                mejor_solucion, mejor_fitness_global = mejor_vecino, mejor_fitness_vecino
                
    return mejor_solucion


def ejecutar_optimizacion_completa(
    datos_base, 
    costo_viaje, 
    ingreso_entrega, 
    penalizacion_dns,
    tamano_poblacion=50, 
    probabilidad_mutacion=0.2, 
    n_generaciones=20, 
    ts_iteraciones=10, 
    ts_tamano_lista=7
):
    """
    Ejecuta el algoritmo HGA-TS completo con un conjunto de parámetros
    y devuelve las métricas clave del resultado.
    """
    
    producer_df, consumer_df, puertos_df, params = datos_base
    consumers_list = consumer_df.to_dict('records')
    
    n_barcos = params['N_BARCOS']
    capacidad_barco = params['CAPACIDAD_BARCO']

    poblacion = [crear_individuo_inteligente(producer_df, consumer_df, n_barcos, capacidad_barco) for _ in range(tamano_poblacion)]
    for ind in poblacion:
        evaluar_fitness(ind, puertos_df, consumers_list, costo_viaje, ingreso_entrega, penalizacion_dns)

    mejor_fitness_global = -float('inf')
    mejor_individuo_global = None

    for gen in range(n_generaciones):
        padres = seleccion_por_ruleta(poblacion)
        
        nueva_poblacion = []
        for i in range(0, tamano_poblacion, 2):
            hijo1, hijo2 = cruzamiento(padres[i], padres[i+1], n_barcos)
            nueva_poblacion.extend([mutacion(hijo1, probabilidad_mutacion), mutacion(hijo2, probabilidad_mutacion)])
        
        for ind in nueva_poblacion:
            evaluar_fitness(ind, puertos_df, consumers_list, costo_viaje, ingreso_entrega, penalizacion_dns)
            
        mejor_de_generacion = max(poblacion, key=lambda x: x['fitness'])
        
        mejor_refinado = busqueda_tabu(
            mejor_de_generacion, puertos_df, consumers_list, 
            costo_viaje, ingreso_entrega, penalizacion_dns,
            ts_iteraciones, ts_tamano_lista
        )
        
        peor_nuevo_idx = min(range(len(nueva_poblacion)), key=lambda i: nueva_poblacion[i]['fitness'])
        if mejor_refinado['fitness'] > nueva_poblacion[peor_nuevo_idx]['fitness']:
            nueva_poblacion[peor_nuevo_idx] = mejor_refinado
        
        poblacion = nueva_poblacion
        
        mejor_fitness_actual = max(ind['fitness'] for ind in poblacion)
        if mejor_fitness_actual > mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_actual
            mejor_individuo_global = max(poblacion, key=lambda x: x['fitness'])


    if mejor_individuo_global is None:
        mejor_individuo_global = max(poblacion, key=lambda x: x['fitness'])

    total_entregado = 0
    consumidores_atendidos = set()
    
    for plan_buque in mejor_individuo_global['rutas']:
        for parada in plan_buque['ruta']:
            if parada['tipo'] == 'descarga':
                total_entregado += parada['cantidad']
                consumidores_atendidos.add(parada['puerto_id'])
    
    return {
        'fitness': mejor_individuo_global['fitness'],
        'total_entregado': total_entregado,
        'consumidores_atendidos': len(consumidores_atendidos),
        'n_consumidores_total': len(consumer_df)
    }


def analizar_y_graficar_resultados(results_df, parametro_variado, titulo_base):
    """
    Toma el DataFrame de resultados y genera gráficos de sensibilidad.
    """
    print(f"\n--- Resultados del Análisis: {titulo_base} ---")
    print(results_df)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    fig.suptitle(f'Análisis de Sensibilidad: {titulo_base}', fontsize=16)
    
    ax1.plot(results_df[parametro_variado], results_df['fitness'], marker='o', color='b')
    ax1.set_ylabel('Fitness (Utilidad)')
    ax1.set_title('Impacto en el Fitness Total')
    ax1.grid(True)
    
    ax2.plot(results_df[parametro_variado], results_df['total_entregado'], marker='o', color='g')
    ax2.set_ylabel('Unidades Entregadas')
    ax2.set_title('Impacto en la Cantidad Total Entregada')
    ax2.grid(True)
    
    ax3.plot(results_df[parametro_variado], results_df['consumidores_atendidos'], marker='o', color='r')
    ax3.set_xlabel(parametro_variado)
    ax3.set_ylabel('N° Consumidores Atendidos')
    ax3.set_title('Impacto en Clientes Atendidos')
    ax3.grid(True)
    
    ax3b = ax3.twinx()
    total_consumidores = results_df['n_consumidores_total'].iloc[0]
    ax3b.set_ylim(0, total_consumidores * 1.1)
    ax3.set_ylim(0, total_consumidores * 1.1)
    cobertura = (results_df['consumidores_atendidos'] / total_consumidores) * 100
    ax3b.plot(results_df[parametro_variado], cobertura, marker='x', color='purple', linestyle='--')
    ax3b.set_ylabel('% Cobertura de Clientes')

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    
    datos_cargados = cargar_y_procesar_datos()
    
    if datos_cargados[0] is not None:
        
       
        PARAM_BASE_GA = {
            'tamano_poblacion': 20,       
            'probabilidad_mutacion': 0.2,
            'n_generaciones': 10,     
            'ts_iteraciones': 5,       
            'ts_tamano_lista': 7
        }
        
        COSTO_VIAJE_BASE = 1.0
        INGRESO_ENTREGA_BASE = 1.0
        PENALIZACION_DNS_BASE = 2.0
        
        
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS 1: Variar Penalización por Demanda No Satisfecha (DNS)")
        print("="*80)
        
        penalizaciones_a_probar = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        resultados_penalizacion = []
        
        for penalizacion in penalizaciones_a_probar:
            print(f"  Ejecutando con PENALIZACION_DNS = {penalizacion}...")
            resultado = ejecutar_optimizacion_completa(
                datos_base=datos_cargados,
                costo_viaje=COSTO_VIAJE_BASE,
                ingreso_entrega=INGRESO_ENTREGA_BASE,
                penalizacion_dns=penalizacion,
                **PARAM_BASE_GA
            )
            resultado['penalizacion_dns'] = penalizacion
            resultados_penalizacion.append(resultado)
            
        df_penalizacion = pd.DataFrame(resultados_penalizacion)
        analizar_y_graficar_resultados(df_penalizacion, 'penalizacion_dns', 'Impacto de la Penalización DNS')

        
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS 2: Variar Ingreso por Entrega")
        print("="*80)

        ingresos_a_probar = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        resultados_ingreso = []

        for ingreso in ingresos_a_probar:
            print(f"  Ejecutando con INGRESO_ENTREGA = {ingreso}...")
            resultado = ejecutar_optimizacion_completa(
                datos_base=datos_cargados,
                costo_viaje=COSTO_VIAJE_BASE,
                ingreso_entrega=ingreso, 
                penalizacion_dns=PENALIZACION_DNS_BASE,
                **PARAM_BASE_GA
            )
            resultado['ingreso_entrega'] = ingreso
            resultados_ingreso.append(resultado)

        df_ingreso = pd.DataFrame(resultados_ingreso)
        analizar_y_graficar_resultados(df_ingreso, 'ingreso_entrega', 'Impacto del Ingreso por Entrega')

        
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS 3: Variar Costo de Viaje")
        print("="*80)
        
        costos_a_probar = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        resultados_costo = []

        for costo in costos_a_probar:
            print(f"  Ejecutando con COSTO_VIAJE = {costo}...")
            resultado = ejecutar_optimizacion_completa(
                datos_base=datos_cargados,
                costo_viaje=costo, 
                ingreso_entrega=INGRESO_ENTREGA_BASE,
                penalizacion_dns=PENALIZACION_DNS_BASE,
                **PARAM_BASE_GA
            )
            resultado['costo_viaje'] = costo
            resultados_costo.append(resultado)
            
        df_costo = pd.DataFrame(resultados_costo)
        analizar_y_graficar_resultados(df_costo, 'costo_viaje', 'Impacto del Costo de Viaje')

        print("\n" + "="*80)
        print("--- ANÁLISIS DE SENSIBILIDAD COMPLETADO ---")
        print("="*80)