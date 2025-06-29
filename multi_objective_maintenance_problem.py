from multiprocessing import Pool
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class Unit:
    def __init__(self, number, capacity, i1_cost, i2_cost, i3_cost, i4_cost):
        self.number = number
        self.capacity = capacity
        self.cost = [i1_cost, i2_cost, i3_cost, i4_cost]


# Parameters
number_of_intervals = 4
number_of_units = 7
population_size = 100
number_of_gen = 30
budget = 900
w1, w2 = 0.1, 1
debug = False

units = [Unit(1, 20, 100, 120, 110, 130),
         Unit(2, 15, 90, 95, 100, 105),
         Unit(3, 35, 80, 85, 90, 95),
         Unit(4, 40, 150, 160, 155, 165),
         Unit(5, 15, 70, 75, 80, 85),
         Unit(6, 15, 60, 65, 70, 75),
         Unit(7, 10, 50, 55, 60, 65)]
intervals = [80, 90, 65, 70]


# Objective Functions
def net_reserve_per_interval(x, interval):
    x = x.reshape(number_of_units, number_of_intervals)
    result = sum(units[u].capacity for u in range(number_of_units) if x[u, interval] == 0)
    return result - intervals[interval]


def total_maintenance_cost(x):
    x = x.reshape(number_of_units, number_of_intervals)
    result = 0
    for u in range(0, number_of_units):
        for j in range(0, number_of_intervals):
            result += x[u][j] * units[u].cost[j]
    return result


# Constraint Functions
def maintenance_duration_violation(x):
    x = x.reshape(number_of_units, number_of_intervals)
    violations = 0
    for u in range(number_of_units):
        active_count = np.sum(x[u])
        if active_count < 1 or active_count > 2:
            violations += 1
        if active_count == 2 and u > 1:
            violations += 1
        if active_count == 1 and u < 2:
            violations += 1
        if u < 2 and active_count == 2:
            indices = np.where(x[u] == 1)[0]
            if indices[1] - indices[0] != 1:
                violations += 1
    return violations


def net_reserve_violation(x):
    x = x.reshape(number_of_units, number_of_intervals)
    return sum(3 for i in range(number_of_intervals) if net_reserve_per_interval(x, i) < 0)


def maintenance_cost_violation(x):
    x = x.reshape(number_of_units, number_of_intervals)
    if budget == 0:  # budget not defined
        return 0
    if total_maintenance_cost(x) > budget:
        return 1
    return 0


# Custom Sampling
def generate_individual(_):
    while True:
        individual = np.random.randint(2, size=(number_of_units, number_of_intervals))
        if (maintenance_duration_violation(individual) == 0 and
                net_reserve_violation(individual) == 0 and
                maintenance_cost_violation(individual) == 0):
            return individual.flatten()  # Compress to 1D


class MyCustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        with Pool() as pool:
            individuals = pool.map(generate_individual, range(n_samples))
        return np.array(individuals)


# Problem
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=number_of_units * number_of_intervals, n_obj=2, n_constr=3,
                         xl=np.zeros(number_of_units * number_of_intervals),
                         xu=np.ones(number_of_units * number_of_intervals))

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.reshape(-1, number_of_units, number_of_intervals)

        f1 = np.array([min(net_reserve_per_interval(x, i) for i in range(number_of_intervals)) for x in X])
        f2 = np.array([total_maintenance_cost(x) for x in X])

        # Constraint violations
        g1 = np.array([maintenance_duration_violation(x) for x in X])
        g2 = np.array([net_reserve_violation(x) for x in X])
        g3 = np.array([maintenance_cost_violation(x) for x in X])

        # Penalty
        penalty = (50 * g1 + 2 * g2 + 2 * g3) * 1000
        penalized_f1 = f1 - penalty
        penalized_f2 = f2 + penalty

        out["F"] = np.column_stack([-w1 * penalized_f1, w2 * penalized_f2])  # Negate f1 for maximization
        out["G"] = np.column_stack([g1, g2, g3])

        if debug:
            print("Generation")
            print(f"reserve: {-w1 * penalized_f1}  cost: {w2 * penalized_f2}")
            print(f"Evaluation: g1={g1}, g2={g2}, g3={g3}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# Run
if __name__ == '__main__':
    problem = MyProblem()

    algorithm = NSGA2(pop_size=population_size,
                      sampling=MyCustomSampling(),
                      crossover=SinglePointCrossover(prob=1.0),
                      mutation=BitflipMutation(prob=0.15),
                      eliminate_duplicates=True)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', number_of_gen),
        seed=1,
        verbose=True)

    # Optimal solutions
    X = (res.X > 0.5).astype(int)
    F = res.F
    G = res.G
    data = list(F)

    print("Optimal solutions:")
    for i, (x, g, f) in enumerate(zip(X, G, data), 1):
        f[0] /= -w1
        f[1] /= w2
        print(f"Solution {i}: {''.join(map(str, x))}")
        print(f"  Duration Violations: {g[0]}, Cost Violations : {g[1]}, Reserve Violations: {g[2]}")
        print(f"  Reserve: {f[0]}, Cost: {f[1]}\n")

    F = tuple(data)

    # Plot results
    plot = Scatter()
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
