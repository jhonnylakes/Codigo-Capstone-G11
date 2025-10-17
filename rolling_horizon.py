import pandas as pd
import heuristica_rolling as opt  # Importamos nuestro motor de optimización

def cargar_datos_iniciales():
    # Esta función es una versión simplificada de la de nuestro optimizador.
    # Solo necesita cargar los datos, no necesita procesarlos para el algoritmo.
    try:
        producer_df = pd.read_csv('Datos/producter_data.csv', index_col=0)
        consumer_df = pd.read_csv('Datos/consumer_data.csv', index_col=0)
        # ... (aquí iría el resto de la lógica de carga si es necesaria)
        
        # Leemos los clusters de tu archivo
        nodos_df = pd.read_csv('Datos/nodos_con_clusters_optimizado.csv')
        consumer_df['cluster'] = nodos_df[nodos_df['Tipo'] == 'Consumidor']['cluster'].values

        print("✓ Datos iniciales cargados para la simulación.")
        return producer_df, consumer_df, None, None # Devolvemos None para los que no usamos aquí
    except Exception as e:
        print(f"Error cargando datos en rolling_horizon: {e}")
        return None, None, None, None

def inicializar_estado_sistema(productores, consumidores, n_barcos):
    """Define el estado del sistema para la Semana 1."""
    print("Iniciando estado del sistema para la Semana 1...")
    
    # Estado de los barcos: todos empiezan en el primer productor de la lista.
    estado_barcos = {f'B{i+1}': {'ubicacion': productores.iloc[0]['id']} for i in range(int(n_barcos))}
    
    # Estado de inventarios: empezamos con inventario cero en los consumidores.
    estado_inventarios = {row['id']: 0 for _, row in consumidores.iterrows()}
    
    return estado_barcos, estado_inventarios

def actualizar_estado_sistema(estado_barcos, estado_inventarios, plan_semanal, consumidores):
    """Actualiza la ubicación de los barcos y los inventarios para la siguiente semana."""
    
    # Actualizar ubicación de barcos
    for plan_buque in plan_semanal['rutas']:
        buque_id = plan_buque['buque_id']
        if plan_buque['ruta']:
            # La nueva ubicación es la última parada de la ruta de la semana.
            estado_barcos[buque_id]['ubicacion'] = plan_buque['ruta'][-1]['puerto_id']

    # Actualizar inventarios (simulación simple)
    # 1. Se consume la demanda de la semana
    for cons_id, inv_actual in estado_inventarios.items():
        demanda = consumidores.loc[consumidores['id'] == cons_id, 'demanda'].iloc[0]
        estado_inventarios[cons_id] = max(0, inv_actual - demanda)
        
    # 2. Se realizan las entregas del plan
    for plan_buque in plan_semanal['rutas']:
        for parada in plan_buque['ruta']:
            if parada['tipo'] == 'descarga':
                estado_inventarios[parada['puerto_id']] += parada['cantidad']

    return estado_barcos, estado_inventarios

# --- BUCLE PRINCIPAL DEL HORIZONTE RODANTE ---
if __name__ == "__main__":
    
    # 1. Carga de datos una sola vez al inicio
    producer_df, consumer_df, puertos_df, params = opt.cargar_y_procesar_datos()

    if producer_df is not None:
        
        # 2. Inicializar el estado del sistema para la Semana 1
        estado_barcos, estado_inventarios = inicializar_estado_sistema(producer_df, consumer_df, params['N_BARCOS'])
        
        plan_anual = [] # Aquí guardaremos los resultados de cada semana
        
        # 3. El gran bucle de 52 semanas
        for semana in range(1, 5):
            print("\n" + "="*50)
            print(f"--- Planificando Semana {semana} ---")
            print(f"Estado inicial de barcos: { {k: v['ubicacion'] for k, v in list(estado_barcos.items())[:3]} }...") # Muestra solo los 3 primeros
            
            # 4. Llamar al optimizador para que resuelva la semana actual
            mejor_plan_semanal = opt.ejecutar_optimizacion_semanal(
                producer_df, consumer_df, puertos_df, params, estado_barcos
            )
            
            if mejor_plan_semanal:
                print(f"Plan óptimo encontrado para la Semana {semana} con Fitness: {mejor_plan_semanal['fitness']:.2f}")
                plan_anual.append(mejor_plan_semanal)
                
                # 5. Actualizar el estado del sistema para la siguiente semana
                estado_barcos, estado_inventarios = actualizar_estado_sistema(
                    estado_barcos, estado_inventarios, mejor_plan_semanal, consumer_df
                )
                # imprimer un resumen del estado actualizado
                print(f"Estado actualizado de barcos: { {k: v['ubicacion'] for k, v in list(estado_barcos.items())[:3]} }...") # Muestra solo los 3 primeros
                print(f"Estado actualizado de inventarios: { {k: v for k, v in list(estado_inventarios.items())[:3]} }...") # Muestra solo los
            else:
                print(f"No se encontró una solución válida para la Semana {semana}. Deteniendo simulación.")
                break
        
        print("\n" + "="*50)
        print("--- SIMULACIÓN DE 52 SEMANAS COMPLETADA ---")
        