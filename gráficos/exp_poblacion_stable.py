# exp_poblacion_stable.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heuristica
import random
import time
import gc

# === Corrida única con parámetros "estables" (sin tocar heuristica.py) ===
def run_once(n_gen, producer_df, consumer_df, puertos_df, params,
             pop_size=100, prob_mut=0.10,
             ts_iters=10, ts_tabu_len=7, refine_every=10,
             seed=1234):
    # Semillas
    random.seed(seed); np.random.seed(seed)

    # Seteo de hiperparámetros globales de la heurística
    heuristica.TAMANO_POBLACION = pop_size
    heuristica.PROBABILIDAD_MUTACION = prob_mut
    heuristica.TS_ITERACIONES = ts_iters
    heuristica.TS_TAMANO_LISTA = ts_tabu_len

    consumers_list = consumer_df.to_dict('records')

    # 1) Población inicial
    poblacion = [
        heuristica.crear_individuo_inteligente(
            producer_df, consumer_df, params['N_BARCOS'], params['CAPACIDAD_BARCO']
        )
        for _ in range(heuristica.TAMANO_POBLACION)
    ]
    for ind in poblacion:
        heuristica.evaluar_fitness(ind, puertos_df, consumers_list)
    mejor_global = max(ind['fitness'] for ind in poblacion)

    # 2) Evolución con ELITISMO (k=2) + Tabú periódico
    for gen in range(n_gen):
        padres = heuristica.seleccion_por_ruleta(poblacion)

        nueva = []
        for i in range(0, heuristica.TAMANO_POBLACION, 2):
            hijo1, hijo2 = heuristica.cruzamiento(padres[i], padres[i+1], params['N_BARCOS'])
            nueva.extend([heuristica.mutacion(hijo1), heuristica.mutacion(hijo2)])

        # Elitismo: conservar top-2 de la población anterior
        elite = sorted(poblacion, key=lambda x: x['fitness'], reverse=True)[:2]

        for ind in nueva:
            heuristica.evaluar_fitness(ind, puertos_df, consumers_list)

        # Reemplaza los 2 peores por la élite si mejora
        worst_order = sorted(range(len(nueva)), key=lambda k: nueva[k]['fitness'])
        for k in range(min(2, len(elite))):
            if elite[k]['fitness'] > nueva[worst_order[k]]['fitness']:
                nueva[worst_order[k]] = elite[k]

        # Búsqueda Tabú sobre el mejor de la generación (cada refine_every gens)
        if refine_every and ((gen + 1) % refine_every == 0 or gen == n_gen - 1):
            best_gen = max(nueva, key=lambda x: x['fitness'])
            refinado = heuristica.busqueda_tabu(best_gen, puertos_df, consumers_list)
            worst_idx = min(range(len(nueva)), key=lambda i: nueva[i]['fitness'])
            if refinado['fitness'] > nueva[worst_idx]['fitness']:
                nueva[worst_idx] = refinado

        poblacion = nueva
        mejor_gen = max(ind['fitness'] for ind in poblacion)
        mejor_global = max(mejor_global, mejor_gen)

    return mejor_global


def main():
    plt.close('all')

    # === GRID de tamaños de población (manteniendo los otros hiperparámetros fijos) ===
    POBLACIONES = [10, 20, 50, 80, 120, 200]
    N_GENERACIONES_FIJO = 80      # mismo criterio estable que usamos en exp_gen_stable
    REPETICIONES = 7
    BASE_SEED = 4242

    # Hiperparámetros "estables" compartidos
    PROB_MUT = 0.10
    TS_ITERS = 10
    TS_TAM_LISTA = 7
    REFINE_EVERY = 10

    # === Carga de datos (una sola vez) ===
    producer_df, consumer_df, puertos_df, params = heuristica.cargar_y_procesar_datos()
    if producer_df is None:
        print("No se pudieron cargar los datos.")
        return

    resultados = []
    print("\n🚀 Experimento (estable): fitness vs tamaño de población\n")

    for pop in POBLACIONES:
        fitness_runs, tiempos = [], []
        print(f"\n▶️ Tamaño de población = {pop}")
        for rep in range(REPETICIONES):
            seed = BASE_SEED + 1000*rep + pop
            t0 = time.time()
            best = run_once(
                N_GENERACIONES_FIJO, producer_df, consumer_df, puertos_df, params,
                pop_size=pop, prob_mut=PROB_MUT,
                ts_iters=TS_ITERS, ts_tabu_len=TS_TAM_LISTA, refine_every=REFINE_EVERY,
                seed=seed
            )
            t1 = time.time()
            fitness_runs.append(best)
            tiempos.append(t1 - t0)
            print(f"   · rep {rep+1}/{REPETICIONES}: best={best:,.2f} | {t1-t0:.1f}s")
            gc.collect()

        row = {
            "tamano_poblacion": pop,
            "fitness_promedio": float(np.mean(fitness_runs)),
            "fitness_desv": float(np.std(fitness_runs)),
            "fitness_mediana": float(np.median(fitness_runs)),
            "fitness_q25": float(np.percentile(fitness_runs, 25)),
            "fitness_q75": float(np.percentile(fitness_runs, 75)),
            "tiempo_promedio": float(np.mean(tiempos))
        }
        resultados.append(row)
        print(f"   ➜ media={row['fitness_promedio']:,.0f} | mediana={row['fitness_mediana']:,.0f} "
              f"| IQR=({row['fitness_q25']:,.0f},{row['fitness_q75']:,.0f}) | t={row['tiempo_promedio']:.1f}s")

    # === Resultados y exportación ===
    df = pd.DataFrame(resultados).sort_values("tamano_poblacion")
    print("\n📊 Resultados:\n")
    print(df.to_string(index=False))

    df.to_csv("resultados_exp_poblacion_stable.csv", index=False)
    print("\n💾 Guardado en resultados_exp_poblacion_stable.csv")

    # === Gráfico 1: media ± std
    plt.figure(figsize=(8,6))
    plt.errorbar(df["tamano_poblacion"], df["fitness_promedio"],
                 yerr=df["fitness_desv"], fmt='-o', capsize=5)
    plt.title("Fitness promedio vs Tamaño de población (± desvío)")
    plt.xlabel("Tamaño de población")
    plt.ylabel("Fitness (más alto es mejor)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === Gráfico 2: mediana con banda IQR
    plt.figure(figsize=(8,6))
    x = df["tamano_poblacion"].values
    y_med = df["fitness_mediana"].values
    y_q25 = df["fitness_q25"].values
    y_q75 = df["fitness_q75"].values
    plt.plot(x, y_med, '-o', label='Mediana')
    plt.fill_between(x, y_q25, y_q75, alpha=0.2, label='IQR (25–75%)')
    plt.title("Sensibilidad del fitness al tamaño de población (Mediana + IQR)")
    plt.xlabel("Tamaño de población")
    plt.ylabel("Fitness (más alto es mejor)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === Gráfico 3: tiempo vs fitness (promedios)
    plt.figure(figsize=(8,6))
    plt.plot(df["tiempo_promedio"], df["fitness_promedio"], 'o-')
    plt.title("Trade-off tiempo vs fitness (promedios)")
    plt.xlabel("Tiempo promedio (s)")
    plt.ylabel("Fitness promedio")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
