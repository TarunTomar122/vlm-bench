# Formal Method: Fixed-Budget Evolutionary Route Search

This document is the paper-ready specification of the implemented route search. The source of truth
is `src/vlm_bench/robust_search.py`, the execution procedure is
`scripts/run_robust_route_search.py`, and frozen run values are in `configs/`.

## 1. Route Intervention

Let a vision encoder contain `L` transformer blocks `F_0, ..., F_{L-1}`. A route is a subset

\[
S \subseteq \mathcal{B}=\{0,\ldots,L-1\}, \qquad |S|=K.
\]

For hidden state `h_l`, route execution is

\[
h_{l+1}=\begin{cases}
h_l, & l\in S,\\
F_l(h_l), & l\notin S.
\end{cases}
\]

Thus, `S` encodes blocks to skip, not blocks to retain. All candidates compared at one budget have
the same cardinality `K`. Search changes neither model weights nor the language decoder.

## 2. Paired Source-Balanced Measurements

Let `c` index visual capability, `d` index dataset source, and `I_{c,d}` be the examples in that
cell. For deterministic correctness indicator `y_i(S)` under route `S`, define

\[
A^S_{c,d}=\frac{1}{|I_{c,d}|}\sum_{i\in I_{c,d}} y_i(S).
\]

Every candidate is paired with full-model predictions on exactly the same examples. Its cell-level
accuracy drop, in percentage points, is

\[
\Delta_{c,d}(S)=100\left(A^0_{c,d}-A^S_{c,d}\right).
\]

All objectives are minimized. Positive values indicate damage from skipping; negative values mean
the route outperformed the full model in that cell. Aggregation operates on cell drops rather than
individual rows, giving each capability-source cell equal weight.

## 3. One Shared Route Objective

Let `G` be all capability-source cells in the relevant scout or evaluation partition. Define

\[
\mu(S)=\frac{1}{|G|}\sum_{g\in G}\Delta_g(S),
\]

\[
w(S)=\max_{g\in G}\Delta_g(S),
\]

\[
\sigma(S)=\sqrt{\frac{1}{|G|}\sum_{g\in G}\left(\Delta_g(S)-\mu(S)\right)^2}.
\]

Evolutionary survival uses the minimization vector

\[
q_{shared}(S)=\left(\mu(S),w(S),\sigma(S)\right).
\]

Development and final selection use the frozen scalar loss

\[
L_{shared}(S)=0.50\mu(S)+0.30w(S)+0.20\sigma(S).
\]

This loss rewards low average damage while penalizing one badly damaged source and inconsistent
behavior across sources.

## 4. Capability-Specific Route Objective

For target capability `t`, let `T_t` be its source cells and `C_t=G\setminus T_t` be every
non-target capability-source cell. Define

\[
\mu_t(S)=\frac{1}{|T_t|}\sum_{g\in T_t}\Delta_g(S),
\qquad
w_t(S)=\max_{g\in T_t}\Delta_g(S),
\]

\[
\sigma_t(S)=\sqrt{\frac{1}{|T_t|}\sum_{g\in T_t}
\left(\Delta_g(S)-\mu_t(S)\right)^2},
\]

and collateral damage

\[
\kappa_t(S)=\frac{1}{|C_t|}\sum_{g\in C_t}\Delta_g(S).
\]

Evolutionary survival minimizes

\[
q_t(S)=\left(\mu_t(S),w_t(S),\kappa_t(S),\sigma_t(S)\right),
\]

while development and final selection minimize

\[
L_t(S)=0.45\mu_t(S)+0.30w_t(S)+0.15\kappa_t(S)+0.10\sigma_t(S).
\]

The collateral term prevents search from freely destroying all other capabilities to improve the
target capability. It is a soft penalty, not a hard constraint.

## 5. Evolutionary Operators

### Initialization

For population size `P`, initialization adds at most `P/2` unique prior routes. Priors come from
the least-damaging single-block ranking and earlier combined-route searches, resized to the current
K. Remaining positions are deterministic pseudo-random K-subsets of the allowed block pool.

### Pareto survival

Candidate `S_a` dominates `S_b` when every component of its matching objective vector is no worse
and at least one is strictly better:

\[
S_a \prec S_b
\iff
\left[\forall j, q_j(S_a)\le q_j(S_b)\right]
\land
\left[\exists j, q_j(S_a)<q_j(S_b)\right].
\]

Candidates are partitioned into nondominated fronts. Fronts are accepted in order until half the
population survives. If only part of a front fits, candidates are ordered by descending NSGA-II
crowding distance, ascending scalar loss, and lexicographic route order. This is elitist: survivors
are copied directly into the next generation.

Before reproduction, survivors are ordered by ascending scalar loss and then lexicographic route
order. If the ordered survivor list is `R=(R_0,...,R_{m-1})`, offspring attempt `i` pairs

\[
A_i=R_{i\bmod m},
\qquad
B_i=R_{(5i+1)\bmod m}.
\]

This is deterministic parent scheduling, not roulette-wheel or tournament selection. Fitness acts
through Pareto survival and survivor ordering.

### Fixed-K crossover

For same-size parents `A` and `B`, the child first retains their intersection. It then takes a
seeded deterministic subset of their non-shared union:

\[
C=(A\cap B)\cup U,
\quad
U\subseteq (A\cup B)\setminus(A\cap B),
\quad
|U|=K-|A\cap B|.
\]

Shared parental choices are therefore always inherited and the child remains a K-block route.

### One-swap mutation

Every newly crossed child is mutated once. For included block `r` and excluded allowed block `a`,

\[
M(C)=(C\setminus\{r\})\cup\{a\},
\qquad r\in C,\quad a\in\mathcal{B}\setminus C.
\]

Seeded SHA-256 ordering chooses `r` and `a`, making the run reproducible and independent of Python
set iteration. Duplicate children are rejected. If repeated crossover-mutation attempts cannot fill
the population, a deterministic random K-subset is inserted.

For example, K4 parents `{2,5,8,11}` and `{2,6,8,14}` force crossover to retain `{2,8}`. It selects
two positions from `{5,6,11,14}`, after which mutation removes one of the four child positions and
adds one allowed position outside the child. There is no crossover or mutation probability: every
offspring receives both operations exactly once, while elite survivors receive neither.

## 6. Search and Freezing Procedure

```text
for each route family, skip budget K, and seed:
    P <- initialize(priors, random K-subsets)
    repeat for the frozen number of evaluated generations:
        evaluate every S in P on the source-balanced scout
        survivors <- ParetoSelect(P, P / 2)
        P <- survivors + unique mutate(crossover(parent pairs))
    retain at most two Pareto finalists from this seed

deduplicate finalists across three seeds
evaluate all finalists on the 300-row development set
advance at most three routes with lowest scalar loss
evaluate those routes on all 876 selection rows
freeze the route with lowest scalar loss
break exact ties by lexicographic block order
```

Qwen2.5-VL used `P=16`, three evaluated generations including the initial population, and seeds
`20260715`, `20260716`, and `20260717` for `K in {4,6,8}`. The lean SmolVLM2 replication used
`P=12`, two evaluated generations, the same three seeds, and `K=4`. A route found at one K is not
automatically reused at another K.

## 7. What This Algorithm Is and Is Not

It is a seeded, fixed-cardinality subset genetic algorithm with NSGA-II-style multi-objective
survival and scalar finalist ranking. It searches interactions among skipped blocks that independent
one-block ranking cannot represent.

It is not a learned per-question router, reinforcement learning, gradient-based architecture search,
or proof of a globally optimal subset. The capability-specific policy assumes that a capability
label is already known and selects one frozen route for that label.
