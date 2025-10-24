
import pandas as pd
import hga_ts_optimizer as opt  

def inicializar_estado_sistema(productores, consumidores, n_barcos):
    """Define el estado del sistema para la Semana 1."""
    print("Iniciando estado del sistema para la Semana 1...")
    
    estado_barcos = {f'B{i+1}': {'ubicacion': productores.iloc[0]['id'], 'carga_a_bordo': 0} for i in range(int(n_barcos))}
    
    estado_inventarios = {row['id']: 0 for _, row in consumidores.iterrows()}
    
    return estado_barcos, estado_inventarios

def actualizar_estado_sistema(estado_barcos, estado_inventarios, plan_semanal, consumidores_df):
    """Actualiza la ubicación, carga de barcos y el inventario de los puertos."""
    
    for plan_buque in plan_semanal['rutas']:
        buque_id = plan_buque['buque_id']
        ruta = plan_buque['ruta']
        if not ruta: continue
        estado_barcos[buque_id]['ubicacion'] = ruta[-1]['puerto_id']
        carga_final = estado_barcos[buque_id].get('carga_a_bordo', 0)
        for parada in ruta:
            if parada['tipo'] == 'carga': carga_final += parada['cantidad']
            else: carga_final -= parada['cantidad']
        estado_barcos[buque_id]['carga_a_bordo'] = carga_final


    for plan_buque in plan_semanal['rutas']:
        for parada in plan_buque['ruta']:
            if parada['tipo'] == 'descarga':
                estado_inventarios[parada['puerto_id']] += parada['cantidad']

    for cons_id, inv_actual in estado_inventarios.items():
        demanda_semanal = consumidores_df.loc[consumidores_df['id'] == cons_id, 'demanda'].iloc[0]
        estado_inventarios[cons_id] = max(0, inv_actual - demanda_semanal)

    return estado_barcos, estado_inventarios

if __name__ == "__main__":
    
    producer_df, consumer_df, puertos_df, params = opt.cargar_y_procesar_datos()

    if producer_df is not None:
        
        estado_barcos, estado_inventarios = inicializar_estado_sistema(producer_df, consumer_df, params['N_BARCOS'])
        
        plan_anual = []
        
        for semana in range(1, 53): 
            print("\n" + "="*50)
            print(f"--- Planificando Semana {semana} ---")
            
            mejor_plan_semanal = opt.ejecutar_optimizacion_semanal(
                producer_df, consumer_df, puertos_df, params, estado_barcos, estado_inventarios
            )
            
            if mejor_plan_semanal:
                print(f"Plan óptimo encontrado para la Semana {semana} con Fitness: {mejor_plan_semanal['fitness']:.2f}")
                plan_anual.append(mejor_plan_semanal)
                
                estado_barcos, estado_inventarios = actualizar_estado_sistema(
                    estado_barcos, estado_inventarios, mejor_plan_semanal, consumer_df
                )
                print(f"Estado actualizado de barcos e inventarios para la siguiente semana.")
                print("\nInventario final de consumidores (primeros 10):")
                for cons_id, inv in list(estado_inventarios.items())[:10]:
                    print(f"  - Consumidor {cons_id}: {inv} unidades")

                print("\nEstado final de barcos (primeros 3):")
                for i in range(1, 4):
                    buque_id = f'B{i}'
                    print(f"  - {buque_id}: Ubicación='{estado_barcos[buque_id]['ubicacion']}', Carga={estado_barcos[buque_id]['carga_a_bordo']:.0f}")
                
            else:
                print(f"No se encontró una solución válida para la Semana {semana}. Deteniendo simulación.")
                break
        
        print("\n" + "="*50)
        print("--- SIMULACIÓN COMPLETADA ---")
        