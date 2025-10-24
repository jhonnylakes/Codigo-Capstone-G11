import pandas as pd
import numpy as np
import pyomo.environ as pyo
from itertools import product


instance = 's'  
base_path = f'datos/{instance}/'

prod_df = pd.read_csv(base_path + 'producter_data.csv')
cons_df = pd.read_csv(base_path + 'consumer_data.csv')
ship_df = pd.read_csv(base_path + 'additional_data.csv')

# --- Limpieza / formateo coordenadas ---
prod_df["x"] = prod_df["Location"].apply(lambda s: float(s.strip("()").split(",")[0]))
prod_df["y"] = prod_df["Location"].apply(lambda s: float(s.strip("()").split(",")[1]))
cons_df["x"] = cons_df["Location"].apply(lambda s: float(s.strip("()").split(",")[0]))
cons_df["y"] = cons_df["Location"].apply(lambda s: float(s.strip("()").split(",")[1]))



P = [f"P{i+1}" for i in range(len(prod_df))]
C = [f"C{i+1}" for i in range(len(cons_df))]
N = P + C
B = [f"B{int(i)}" for i in ship_df["Ship"].tolist()]
T = range(1, 6)  

CapP = dict(zip(P, prod_df["Offer"]))
CapC = dict(zip(C, cons_df["Capacity"]))
demand = dict(zip(C, cons_df["Demand"]))
Park = {n: 1 for n in N} 
h = {c: 0.02 for c in C}
g = float(ship_df["Inventory_Cost"].iloc[0])
CapB = float(ship_df["Ship_Capacity"].iloc[0])
K = 1200  
p = {c: 8 for c in C}


coords = {}
for i, row in prod_df.iterrows():
    coords[P[i]] = (row["x"], row["y"])
for i, row in cons_df.iterrows():
    coords[C[i]] = (row["x"], row["y"])

A = []
dist = {}
for i, j in product(N, N):
    if i != j:
        A.append((i, j))
        xi, yi = coords[i]
        xj, yj = coords[j]
        dist[(i, j)] = round(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)


m = pyo.ConcreteModel()
m.A = pyo.Set(initialize=A, dimen=2)
m.N = pyo.Set(initialize=N)
m.P = pyo.Set(initialize=P)
m.C = pyo.Set(initialize=C)
m.B = pyo.Set(initialize=B)
m.T = pyo.Set(initialize=T)


m.x = pyo.Var(m.A, m.B, m.T, within=pyo.Binary)
m.z = pyo.Var(m.N, m.B, m.T, within=pyo.Binary)
m.r = pyo.Var(m.N, m.B, m.T, within=pyo.Binary)
m.L = pyo.Var(m.A, m.B, m.T, within=pyo.NonNegativeReals)
m.H = pyo.Var(m.N, m.B, m.T, within=pyo.NonNegativeReals)
m.U = pyo.Var(m.B, m.T, within=pyo.NonNegativeReals)
m.l = pyo.Var(m.N, m.B, m.T, within=pyo.Reals)
m.y = pyo.Var(m.P, m.T, within=pyo.NonNegativeReals)
m.c = pyo.Var(m.C, m.T, within=pyo.NonNegativeReals)
m.I = pyo.Var(m.C, m.T, within=pyo.NonNegativeReals)
m.S = pyo.Var(m.C, m.T, within=pyo.NonNegativeReals)


m.dist = pyo.Param(m.A, initialize=dist)


def obj_rule(m):
    c_trans = sum(m.dist[i,j] * m.x[i,j,b,t] for (i,j) in m.A for b in m.B for t in m.T)
    c_inv = sum(h[c] * m.I[c,t] for c in m.C for t in m.T)
    c_ship = sum(g * m.U[b,t] for b in m.B for t in m.T)
    c_loss = sum(p[c] * m.S[c,t] for c in m.C for t in m.T)
    return c_trans + c_inv + c_ship + c_loss
m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

m.CapTramo = pyo.Constraint(m.A, m.B, m.T, rule=lambda m,i,j,b,t: m.L[i,j,b,t] <= CapB * m.x[i,j,b,t])


def bal_bordo(m,i,b,t):
    if t == 1:
        return sum(m.L[i,j,b,t] for j in m.N if (i,j) in m.A) - \
               sum(m.L[j,i,b,t] for j in m.N if (j,i) in m.A) == m.l[i,b,t] - m.H[i,b,t]
    else:
        return sum(m.L[i,j,b,t] for j in m.N if (i,j) in m.A) - \
               sum(m.L[j,i,b,t] for j in m.N if (j,i) in m.A) == m.l[i,b,t] + m.H[i,b,t-1] - m.H[i,b,t]
m.BalBordo = pyo.Constraint(m.N, m.B, m.T, rule=bal_bordo)


m.CapH = pyo.Constraint(m.N, m.B, m.T, rule=lambda m,i,b,t: m.H[i,b,t] <= CapB * m.r[i,b,t])
m.UBal = pyo.Constraint(m.B, m.T, rule=lambda m,b,t: m.U[b,t] == sum(m.H[i,b,t] for i in m.N))


def link_rule(m,i,t):
    if i in P:
        return sum(m.l[i,b,t] for b in m.B) == m.y[i,t]
    elif i in C:
        return -sum(m.l[i,b,t] for b in m.B) == m.c[i,t]
    else:
        return sum(m.l[i,b,t] for b in m.B) == 0
m.Link = pyo.Constraint(m.N, m.T, rule=link_rule)


m.CapProd = pyo.Constraint(m.P, m.T, rule=lambda m,i,t: m.y[i,t] <= CapP[i])


def inv_rule(m,c,t):
    if t == 1:
        return m.I[c,t] == m.c[c,t] - demand[c] + m.S[c,t]
    else:
        return m.I[c,t] == m.I[c,t-1] + m.c[c,t] - demand[c] + m.S[c,t]
m.InvCons = pyo.Constraint(m.C, m.T, rule=inv_rule)
m.CapInv = pyo.Constraint(m.C, m.T, rule=lambda m,c,t: m.I[c,t] <= CapC[c])

m.DistCap = pyo.Constraint(m.B, m.T, rule=lambda m,b,t: sum(m.dist[i,j]*m.x[i,j,b,t] for (i,j) in m.A) <= K)


solver = pyo.SolverFactory('gurobi')
solver.solve(m, tee=True)


def v(expr): return pyo.value(expr)

C_trans = sum(v(m.dist[i,j]) * v(m.x[i,j,b,t]) for (i,j) in A for b in B for t in T)
C_inv = sum(h[c]*v(m.I[c,t]) for c in C for t in T)
C_ship = sum(g*v(m.U[b,t]) for b in B for t in T)
C_loss = sum(p[c]*v(m.S[c,t]) for c in C for t in T)
CostoTotal = C_trans + C_inv + C_ship + C_loss

total_demanda = sum(demand[c]*len(T) for c in C)
total_satisfecha = sum(v(m.c[c,t]) for c in C for t in T)
fill_rate = 100 * total_satisfecha / total_demanda
unidades_entregadas = total_satisfecha / len(T)
consumidores_atendidos = np.mean([len([c for c in C if v(m.c[c,t])>0]) for t in T])
uso_flota = np.mean([sum(v(m.U[b,t])/CapB for b in B)/len(B) for t in T]) * 100

print("\n====== RESULTADOS ======")
print(f"Costo total: {CostoTotal:,.2f}")
print(f"  Costo transporte: {C_trans:,.2f}")
print(f"  Costo inventario consumidores: {C_inv:,.2f}")
print(f"  Costo inventario a bordo: {C_ship:,.2f}")
print(f"  Costo por demanda no servida: {C_loss:,.2f}")
print(f"% Demanda satisfecha promedio semanal: {fill_rate:.1f}%")
print(f"% Utilización de flota promedio semanal: {uso_flota:.1f}%")
print(f"Nº Unidades entregadas promedio semanal: {unidades_entregadas:.1f}")
print(f"Nº Consumidores atendidos promedio semanal: {consumidores_atendidos:.1f}")
print("=========================\n")
