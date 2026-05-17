# Analytic Hierarchy Process (AHP)

A Python implementation of the **Analytic Hierarchy Process** — a method of multi-criteria, multi-objective decision making developed by Thomas Saaty.

The program ranks a set of alternatives against several criteria using pairwise comparison matrices (PCMs), checks them for consistency, and synthesises local priorities into a global ranking.

---

## Algorithm

The hierarchy is **three-level**:

- **Level 1** — Global goal.
- **Level 2** — Criteria $C_i,\ i = \overline{1, n}$.
- **Level 3** — Alternatives $A_k,\ k = \overline{1, m}$.

### Step 0. Input

Two inputs are supplied:

1. Criteria pairwise comparison matrix $A = (a_{ij})_{i,j \in \overline{1,n}}$, $a_{ij} > 0$, $a_{ji} = 1/a_{ij}$, $a_{ii} = 1$, built in the **fundamental Saaty scale**.
2. Alternative PCMs $A^{(k)},\ k = \overline{1, n}$ — one per criterion — or a raw performance table from which they are derived by the ratio rule $a_{ij} = \mathrm{score}_i / \mathrm{score}_j$ (for "$\to \max$" criteria; for "$\to \min$" criteria the ratio is inverted).

### Step 1. Local priority vector (RGMM)

The Row Geometric Mean Method is used to obtain the priority vector of every PCM.

Unnormalised weights:

$$
v_i = \left(\prod_{j=1}^{n} a_{ij}\right)^{1/n}, \qquad i = \overline{1,n}.
$$

Normalised weights:

$$
w_i = \frac{v_i}{\displaystyle\sum_{k=1}^{n} v_k}, \qquad \sum_{i=1}^{n} w_i = 1.
$$

### Step 2. Consistency check

Column sums of the PCM:

$$
\sigma_j = \sum_{i=1}^{n} a_{ij}, \qquad j = \overline{1,n}.
$$

Principal eigenvalue (approximation):

$$
\lambda_{\max} = \sum_{i=1}^{n} \sigma_i \cdot w_i.
$$

Consistency Index:

$$
C\!I = \frac{\lambda_{\max} - n}{n - 1}.
$$

Consistency Ratio:

$$
C\!R = \frac{C\!I}{M\!RC\!I},
$$

where $M\!RC\!I$ is the random consistency index from Saaty's reference table (function of $n$).

The PCM is regarded as **consistent** when

$$
C\!R \le \tau(n),
$$

where the threshold is $\tau(3) = 0.05$, $\tau(4) = 0.08$, $\tau(n \ge 5) = 0.10$.

### Step 3. Automatic improvement of consistency

If $C\!R > \tau(n)$, the PCM is iteratively pulled towards a perfectly consistent matrix $B = (w_i / w_j)$ using one of two transformations with smoothing parameter $\alpha \in (0, 1)$.

**Weighted Arithmetic Mean (WAM):**

$$
a_{ij}^{(t+1)} =
\begin{cases}
\alpha \cdot a_{ij}^{(t)} + (1-\alpha)\cdot \dfrac{w_i}{w_j}, & j > i, \\[6pt]
1, & i = j, \\[6pt]
1 / a_{ji}^{(t+1)}, & j < i.
\end{cases}
$$

**Weighted Geometric Mean (WGM):**

$$
a_{ij}^{(t+1)} =
\begin{cases}
\left(a_{ij}^{(t)}\right)^{\alpha} \cdot \left(\dfrac{w_i}{w_j}\right)^{1-\alpha}, & j > i, \\[6pt]
1, & i = j, \\[6pt]
1 / a_{ji}^{(t+1)}, & j < i.
\end{cases}
$$

After every iteration weights are recomputed and the modification is accepted only if the perturbation stays within tolerances:

$$
\delta^{(t+1)} = \max_{i,j} \left| a_{ij}^{(t+1)} - a_{ij}^{(t)} \right| \le 0.2,
$$

$$
\sigma^{(t+1)} = \frac{1}{n}\sqrt{\sum_{i=1}^{n}\sum_{j=1}^{n} \left(a_{ij}^{(t+1)} - a_{ij}^{(t)}\right)^2} \le 0.1.
$$

The procedure repeats until $C\!R \le \tau(n)$ or the perturbation bounds are exceeded.

### Step 4. Local priorities of alternatives

Steps 1–3 are repeated for each alternative PCM $A^{(k)}$, $k = \overline{1, n}$, yielding the local-weights vector

$$
w^{(k)} = \big(w^{(k)}_1,\ w^{(k)}_2,\ \ldots,\ w^{(k)}_m\big), \qquad \sum_{i=1}^{m} w^{(k)}_i = 1.
$$

### Step 5. Global synthesis (distributive method)

Global priority of alternative $A_i$ is the weighted sum of its local priorities across all criteria:

$$
w_i^{\mathrm{glob}} = \sum_{j=1}^{n} w_{C_j} \cdot w_{ij}, \qquad i = \overline{1, m},
$$

where $w_{C_j}$ is the global weight of criterion $C_j$ and $w_{ij}$ is the local weight of alternative $A_i$ under criterion $C_j$.

The final ranking is obtained by sorting alternatives by $w_i^{\mathrm{glob}}$ in descending order; the largest value identifies the best alternative.

---

## Project structure

```
.
├── main.py        # AHP implementation
├── LICENSE
├── .gitignore
└── README.md
```

## Run

```bash
python main.py
```

## Your input

To change the input, change the parameters in main function.
CLI is TBD.