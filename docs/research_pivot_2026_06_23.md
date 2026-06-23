# RelaLeap Research Pivot: Causal Columnar Residual Learning

Date: 2026-06-23

This note records Ben's current scientific direction shift for RelaLeap. It is
intended as an instruction document for the automation loop and as a concise
handoff for outside reviewers.

## Core Shift

Do not abandon the residual-column architecture. Change the scientific
emphasis.

The harness phase has succeeded: the code can train residual parameters,
preserve the frozen base, reproduce local and Colab behavior, constrain
settling, and emit auditable artifacts. But further optimization of
`1e-4`-scale cross-entropy changes is unlikely to test the central claim.

The next central question is:

```tex
\[
\boxed{
\text{Can the residual layer learn causally separable, reusable corrections
with less interference than matched alternatives?}
}
\]
```

Cross-entropy and perplexity remain guardrails. They should no longer be the
sole promotion criterion.

## Interpret Current Results This Way

Temporal-consistency settling is useful mainly as a stability mechanism. Its
repeatability is encouraging, but its small magnitude suggests diminishing
returns from more variants of the same label-free settling objective.

The top-k `2` support-width result is more important, but it is confounded. With
four atoms per column, moving from top-k `1` to top-k `2` roughly doubles active
rank, active parameters, and residual magnitude. Doubling stored columns while
keeping top-k `1` does not. The current result may therefore be an active-rank
bottleneck rather than proof of column cooperation.

The current PC result is not diagnostic enough. A one-site logit-MSE objective
is only a loose PC proxy. It does not test whether iterative latent inference
plus local residual updates improves credit assignment, modularity, retention,
or continual learning. In addition, asking PC to immediately beat supervised CE
on CE is a narrow gate for the broader hypothesis.

Keep a clean conceptual separation:

```tex
\[
\text{training-time supervised HEP}
\neq
\text{test-time label-free latent refinement}.
\]
```

Guided HEP can be legitimate during training. The right test is to train with
supervised HEP where appropriate, then evaluate without labels. Temporal
consistency is a separate label-free refinement mechanism and should be named
and evaluated as such.

## Immediate Priority: Deconfound Support Width

Before adding new architectural machinery, run a tightly controlled support
experiment. Compare at matched active rank, storage, and compute as much as the
current harness allows:

```tex
\[
\begin{array}{lll}
A:& k=1,& 8\text{ atoms per active column},\\
B:& k=2,& 4\text{ atoms per active column, current learned router},\\
C:& k=2,& 4\text{ atoms, random fixed supports},\\
D:& k=2,& 4\text{ atoms, learned or audited supports},\\
E:& \text{dense residual},& \text{matched active rank and FLOPs}.
\end{array}
\]
```

For top-k `2`, run two residual-sum normalizations:

```tex
\[
r = \frac{1}{k}\sum_{c \in S} r_c
\qquad \text{and} \qquad
r = \frac{1}{\sqrt{k}}\sum_{c \in S} r_c.
\]
```

Every run should log:

- active parameter count;
- stored parameter count;
- approximate FLOPs;
- residual norm before and after normalization;
- support identity churn;
- functional churn;
- anchor-task drift;
- ordinary held-out loss and best accepted HEP loss.

This matrix should distinguish:

- active-rank effects;
- residual-scale effects;
- top-k smoothing effects;
- genuine learned column cooperation.

A convincing result is not simply "top-k `2` improves CE." A convincing result
is learned top-k `2` beating random top-k `2`, rank-matched top-k `1`, and a
dense rank/FLOP-matched residual under comparable drift budgets.

## Exhaustively Audit the Support Landscape

At current scale, exact audits are cheap. Do them rather than guessing.

For `N=12` columns, there are `C(12,2)=66` two-column supports. For `N=24`,
there are `C(24,2)=276`.

Evaluate every support on fixed held-out batches. For each example or context,
record router loss and oracle loss:

```tex
\[
R_{\mathrm{oracle}} =
L(S_{\mathrm{router}}) - \min_{|S|=k} L(S).
\]
```

For support `S`, define gain:

```tex
\[
G(S)=L(\varnothing)-L(S).
\]
```

For pairs, define synergy:

```tex
\[
\operatorname{Syn}(a,b)=G(\{a,b\})-G(\{a\})-G(\{b\}).
\]
```

Primary audit outputs:

- oracle-support regret;
- distribution of support losses;
- best support, router support, and one-swap neighbor supports;
- pairwise synergy table;
- redundancy indicators across columns;
- whether local one-swap teachers can recover much of the oracle gap.

Interpretation:

- large oracle regret means the router is the bottleneck;
- similar support performance means columns are redundant;
- a few strong pairs suggest complementary structure;
- strong pair effects with weak singleton effects motivate a composer or
  mediator mechanism.

## Replace Raw Support Churn With Functional Churn

Raw support identity churn is not enough. Large support changes can be harmless
when columns are functionally interchangeable.

Report identity churn:

```tex
\[
C_{\mathrm{id}} = 1-\operatorname{Jaccard}(S_t,S_{t+1}).
\]
```

Report functional churn:

```tex
\[
C_{\mathrm{func}} =
D_{\mathrm{KL}}\left(p(\cdot \mid S_t) \,\|\, p(\cdot \mid S_{t+1})\right).
\]
```

Also report residual-stream churn:

```tex
\[
C_{\mathrm{res}} =
\| U_{S_t}(h)-U_{S_{t+1}}(h)\|.
\]
```

Add routing margin between the `k`-th and `k+1`-st scores. High identity churn,
low functional churn, and tiny routing margins indicate arbitrary tie-breaking
among redundant columns, not necessarily unstable dynamics.

## Make Continual Learning a Primary Assay

The continual-learning test must avoid trivial side channels:

- one shared output head;
- fixed total column capacity;
- fixed active-support budget;
- no external task identifier;
- no manually assigned task-to-column map;
- hidden task boundaries;
- no replay in the primary experiment, or exactly matched replay for every
  method;
- matched storage and active compute across baselines;
- multiple task orders;
- recurring tasks.

A useful stream shape is:

```tex
\[
A \to B \to C \to D \to A \to C.
\]
```

Evaluate every seen task after every phase. For task `i`, define forgetting as:

```tex
\[
F_i = L_i^{\mathrm{final}} - \min_{s \ge i} L_i^{(s)}.
\]
```

Report mean seen-task loss, backward transfer, forward transfer, old-task KL
drift, storage growth, and active compute.

Start with a mechanism-factorized synthetic language benchmark rather than only
Shakespeare domains:

- same vocabulary and output head for all tasks;
- episodes generated by one or two latent rules;
- no task token;
- strong input overlap between tasks;
- rules known only to the evaluator;
- held-out episodes with unseen rule combinations.

Candidate mechanisms: relation inversion, transitive chaining, negation,
attribute substitution, sequence reversal, and token permutation.

## Measure Causal Modularity Directly

For each column `c` and known mechanism `m`, estimate causal impact:

```tex
\[
I_{c,m}=L_m(\operatorname{ablate}(c))-L_m(\operatorname{full}).
\]
```

Normalize positive impacts into a column fingerprint:

```tex
\[
p_{c,m} =
\frac{[I_{c,m}]_+}{\sum_{m'}[I_{c,m'}]_+ + \epsilon}.
\]
```

Define intervention purity:

```tex
\[
P_c = 1-\frac{H(p_c)}{\log M}.
\]
```

Also report:

- necessity: loss increase when a selected column is removed;
- sufficiency: performance using only the selected support;
- selectivity: selected support versus random matched support;
- off-target leakage on unrelated mechanisms;
- pairwise synergy;
- cross-seed role stability.

This is stronger than probing correlations. It asks whether intervening on a
column selectively changes behavior.

## Add a Finite-Update Commutator Assay

Compare update order directly. Let `U_i` be one controlled update on task `i`.

```tex
\[
\theta_{ij}=U_j(U_i(\theta)), \qquad
\theta_{ji}=U_i(U_j(\theta)).
\]
```

Measure parameter and behavioral disagreement:

```tex
\[
C_{ij}^{\mathrm{param}} =
\frac{\|\theta_{ij}-\theta_{ji}\|}
{\|\Delta_i\|\,\|\Delta_j\|+\epsilon},
\]
```

```tex
\[
C_{ij}^{\mathrm{beh}} =
\mathbb{E}_x
D_{\mathrm{KL}}\left(
p_{\theta_{ij}}(\cdot\mid x)
\|
p_{\theta_{ji}}(\cdot\mid x)
\right).
\]
```

Use simple SGD or reset optimizer state so Adam state does not become the
measured phenomenon.

Compare dense residual adapters, ordinary low-rank adapters, random sparse
columns, learned columns, learned columns with a support teacher, and PC-trained
columns. A meaningful result is similar held-out quality with lower commutator
scores and less sequential forgetting.

## Dense-Teacher Residual Distillation

Separate representability from discovery.

First train a dense BP residual adapter `A^*` that clearly improves the target
task. Record its hidden correction:

```tex
\[
r^*(x)=h_{A^*}(x)-h_0(x).
\]
```

Then train the columnar layer to reconstruct that correction:

```tex
\[
L_{\mathrm{distill}} =
\|U_{S(x)}(h_0)-r^*(x)\|^2
+ \lambda_{\mathrm{sparse}} |S(x)|
+ \lambda_{\mathrm{overlap}} L_{\mathrm{overlap}}.
\]
```

Run with ordinary BP through columns, local PC updates, and local PC plus
supervised training-time HEP. This separates:

- the desired residual is not low-rank or columnable;
- columns can represent it, but the router cannot find support;
- representation and routing work, but the PC update cannot learn them.

## Test Real PC/HEP Where It Should Matter

The current one-site setup is too shallow to strongly test PC credit assignment.
Build a modest residual PC graph with 4-8 trainable residual sites or a short
residual tower.

Compare:

```tex
\[
\text{BP columns}, \qquad
\text{vanilla-PC columns}, \qquad
\text{PC+HEP columns}.
\]
```

Use supervised output error for HEP during training. Evaluate with feedforward
inference, optionally followed by separately named label-free temporal
refinement.

## Symbolic Heads as Diagnostics First

Add a simple symbolic mechanism readout only as a diagnostic at first:

```tex
\[
\hat M(x)=H_{\mathrm{sym}}(C_1(x),\ldots,C_N(x)).
\]
```

Initial restrictions:

- train the head after the column model;
- do not send gold mechanism labels into the router;
- do not let the symbolic head write back into the model.

Evaluate whether symbolic predictions remain consistent under causal
interventions. Only after this works should symbolic loss influence the columns,
and only later should symbolic state write back into the loop.

## Revised Promotion Criterion

A variant advances when it:

1. preserves or improves held-out task performance;
2. materially reduces forgetting or update noncommutativity;
3. improves causal-support purity or oracle-support regret;
4. respects matched active-parameter, storage, and compute budgets;
5. preserves anchor-task behavior.

Use target-domain gain versus held-out anchor KL drift, not just a strict
target-example logit-delta threshold. Changing target logits is the point of
adaptation. Collateral change on unrelated anchors is the safety concern.

## Recommended Next Sequence

1. Finish the residual-capacity decision report, but treat mixed local/Colab
   directionality as weak evidence rather than a main research path.
2. Run the active-rank, norm, and compute-matched support-width matrix.
3. Exhaustively map the top-k `1` and top-k `2` support landscape.
4. Add oracle and one-swap support teachers.
5. Run dense-teacher residual distillation to test columnability.
6. Introduce the mechanism-factorized task-free continual-learning stream.
7. Make causal impact, commutator, and forgetting metrics primary.
8. Add a symbolic mechanism probe.
9. Test supervised training-time HEP in a deeper residual PC graph.
10. Return to more elaborate symbolic or looped architectures only after these
    assays show clean modular structure.

The strongest near-term target result is:

> At matched task quality, active parameters, and compute, causally routed
> residual columns exhibit lower update commutators, lower task-free continual
> forgetting, and more selective intervention fingerprints than dense or
> ordinary sparse residual adapters.
