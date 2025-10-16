
import pandas as pd
import numpy as np
import random
import math
import copy

# 1. LECTURA DE DATOS
instance = 's'  
base_path = f'datos/{instance}/'

producers = pd.read_csv(base_path + 'producter_data.csv')
consumers = pd.read_csv(base_path + 'consumer_data.csv')
params = pd.read_csv(base_path + 'additional_data.csv')

# Procesar coordenadas
def parse_loc(loc_str):
    x, y = loc_str.strip('()').split(',')
    return float(x), float(y)

producers['x'], producers['y'] = zip(*producers['Location'].map(parse_loc))
consumers['x'], consumers['y'] = zip(*consumers['Location'].map(parse_loc))

# IDs y sets
producer_ids = list(producers['Producer'])
consumer_ids = list(consumers['Consumer'])
all_nodes = producer_ids + consumer_ids

# Parámetros globales
ship_capacity = params['Ship_Capacity'][0]
inv_cost = params['Inventory_Cost'][0]
cost_transp = 1.0
penalty_short = 10.0
fleet_size = 3  # puede ajustarse según instancia

# ==========================
# 2. MATRIZ DE DISTANCIAS
# ==========================
coords = {row['Producer']: (row['x'], row['y']) for _, row in producers.iterrows()}
coords.update({row['Consumer']: (row['x'], row['y']) for _, row in consumers.iterrows()})

n = len(all_nodes)
dist_matrix = np.zeros((n, n))

for i, ni in enumerate(all_nodes):
    for j, nj in enumerate(all_nodes):
        if i != j:
            xi, yi = coords[ni]
            xj, yj = coords[nj]
            dist_matrix[i, j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)

# ==========================
# 3. MODELO DEL PROBLEMA
# ==========================
class MaritimeProblem:
    def __init__(self):
        self.P = producer_ids
        self.C = consumer_ids
        self.dist = dist_matrix
        self.capacity = ship_capacity
        self.demand = dict(zip(consumers['Consumer'], consumers['Demand']))
        self.offer = dict(zip(producers['Producer'], producers['Offer']))
        self.cost_transp = cost_transp
        self.inv_cost = inv_cost
        self.penalty_short = penalty_short
        self.fleet_size = fleet_size

    def cost_route(self, route):
        cost = 0
        for i in range(len(route) - 1):
            idx_i = all_nodes.index(route[i])
            idx_j = all_nodes.index(route[i + 1])
            cost += self.cost_transp * self.dist[idx_i, idx_j]
        return cost

problem = MaritimeProblem()

# ==========================
# 4. HEURÍSTICA HÍBRIDA GA–TS
# ==========================

def random_solution(problem):
    ports = problem.P + problem.C
    random.shuffle(ports)
    return [ports[i::problem.fleet_size] for i in range(problem.fleet_size)]

def evaluate_solution(problem, solution):
    total_cost = 0
    served = set()
    for route in solution:
        total_cost += problem.cost_route(route)
        for node in route:
            if node in problem.C:
                served.add(node)
    fill_rate = len(served) / len(problem.C)
    return total_cost, fill_rate

def crossover(parent1, parent2):
    child = []
    for r1, r2 in zip(parent1, parent2):
        cut = len(r1)//2
        merged = r1[:cut] + [x for x in r2 if x not in r1[:cut]]
        child.append(merged)
    return child

def mutate(solution, rate=0.1):
    for route in solution:
        if random.random() < rate and len(route) > 2:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    return solution

def tabu_search(problem, initial_solution, max_iter=50, tabu_tenure=10):
    best = copy.deepcopy(initial_solution)
    best_cost, _ = evaluate_solution(problem, best)
    tabu_list = []

    for _ in range(max_iter):
        neighborhood = []
        for route in best:
            if len(route) > 2:
                i, j = random.sample(range(len(route)), 2)
                neighbor = copy.deepcopy(best)
                r_copy = neighbor[0][:]
                r_copy[i], r_copy[j] = r_copy[j], r_copy[i]
                neighbor[0] = r_copy
                neighborhood.append(neighbor)

        for neighbor in neighborhood:
            cost, _ = evaluate_solution(problem, neighbor)
            if cost < best_cost and neighbor not in tabu_list:
                best = neighbor
                best_cost = cost
                tabu_list.append(neighbor)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)

    return best

def hybrid_GA_TS(problem, pop_size=8, generations=10):
    population = [random_solution(problem) for _ in range(pop_size)]
    scores = [evaluate_solution(problem, s) for s in population]
    best = population[np.argmin([c for c, _ in scores])]

    for g in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = random.sample(population, 2)
            child = mutate(crossover(p1, p2))
            child = tabu_search(problem, child, max_iter=20)
            new_pop.append(child)
        population = new_pop
        scores = [evaluate_solution(problem, s) for s in population]
        best = population[np.argmin([c for c, _ in scores])]

    return best, evaluate_solution(problem, best)

# ==========================
# 5. EJECUCIÓN
# ==========================

best_sol, (best_cost, fill_rate) = hybrid_GA_TS(problem, pop_size=10, generations=15)

print("===== RESULTADOS =====")
print(f"Costo total aproximado: {best_cost:.2f}")
print(f"Fill rate: {fill_rate*100:.1f}%")
print("Rutas por buque:")
for i, route in enumerate(best_sol):
    print(f"Buque {i+1}: {route}")
