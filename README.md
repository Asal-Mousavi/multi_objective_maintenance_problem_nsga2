# Multi-Objective Maintenance Scheduling Optimization

This project implements a multi-objective optimization model using evolutionary algorithms to schedule maintenance of power generation units over multiple time intervals. The goal is to **maximize system reserve margins** while **minimizing total maintenance costs**, subject to operational and budgetary constraints.

## Key Features

- **Evolutionary Optimization**: Uses the NSGA-II algorithm from the `pymoo` library for solving multi-objective problems.
- **Custom Sampling**: Ensures that initial populations comply with constraints through multiprocessing-based sampling.
- **Constraint Management**: Enforces maintenance duration limits, reserve requirements, and cost budgets.
- **Visualization**: Plots the Pareto front to illustrate the trade-offs between objectives.

---

##  How It Works

### Problem Description

You have `N` generating units, each with:
- A maximum capacity (MW)
- Maintenance costs per time interval
- Constraints on how long they can undergo maintenance

You also have:
- A power reserve demand per time interval
- A limited maintenance **budget**

The goal is to **schedule which units undergo maintenance during which time intervals**, such that:
1. **Minimum reserve** across all intervals is maximized.
2. **Total maintenance cost** is minimized.
3. Constraints on unit operation and cost are satisfied.

### Objectives

1. **Maximize**: Minimum reserve margin across all time intervals
2. **Minimize**: Total maintenance cost

### Constraints

- Each unit must be in maintenance for **1 or 2 intervals**, depending on its type.
- Reserve margin must meet or exceed the demand in each interval.
- The total maintenance cost must stay within the defined budget.

---

## 🔧 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **pymoo** | Multi-objective optimization library |
| **NumPy** | Array manipulation and numerical operations |
| **matplotlib / pymoo.visualization** | Visualization of Pareto front |
| **multiprocessing** | Parallel sampling of valid solutions |

---

## Algorithm and Components

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

- A robust multi-objective optimization algorithm
- Maintains a diverse Pareto front of optimal solutions
- Uses:
  - **Single Point Crossover**
  - **Bitflip Mutation**
  - **Custom Sampling** to enforce constraint-compliant individuals from the start

### Custom Components

- **Unit Class**: Represents a generating unit and its characteristics
- **Constraint Functions**:
  - `maintenance_duration_violation`
  - `net_reserve_violation`
  - `maintenance_cost_violation`
- **Custom Sampling**: Generates only valid solutions using constraint checks before fitness evaluation
- **Objective Functions**:
  - `net_reserve_per_interval`
  - `total_maintenance_cost`

---

## Output Example

Each solution includes:
- Binary scheduling string (unit × interval)
- Violations summary (duration, reserve, cost)
- Objective values (reserve margin, maintenance cost)

Sample output:
```plaintext
Solution 1: 010001011001...
  Duration Violations: 0, Cost Violations : 0, Reserve Violations: 0
  Reserve: 90, Cost: 745
