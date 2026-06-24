# Contextual Support Router Promotion Gate Satisfaction

- Status: `pass`
- Decision: `satisfy_contextual_support_router_promotion_or_repeat_gate`
- Selected next direction: `contextual_support_router_default_config_update`
- Promote contextual support router default: `True`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`
- Default support width top-k: `2`

## Rationale

The bounded promotion gate now has matching local and real-Chrome Colab evidence on both required settings. In every backend and dataset cell, the contextual MLP support router lowers alpha-0 CE loss and expands support utilization versus the linear top-k-2 router while preserving supervised CE, temporal-clipped HEP, and support_stress_preset: false. Nonzero HEP alphas still do not drive the gain, so the default change should be scoped to support routing.

## Evidence

| Backend | Dataset | Seed | Artifact check | Verdict | Linear alpha-0 | Contextual alpha-0 | Delta | Linear used | Contextual used | Linear supports | Contextual supports | Support-change delta |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local | `tiny_shakespeare_char` | 2 | `pass` | `pass` | 3.15261722 | 1.84800947 | -1.30460775 | 6 | 19 | 9 | 60 | -0.08789062 |
| local | `tiny_shakespeare_word` | 1 | `pass` | `pass` | 3.51685524 | 2.86810708 | -0.64874816 | 16 | 20 | 31 | 50 | -0.33984375 |
| colab | `tiny_shakespeare_char` | 2 | `pass` | `pass` | 3.20794559 | 1.84802544 | -1.35992014 | 12 | 19 | 13 | 60 | -0.02734375 |
| colab | `tiny_shakespeare_word` | 1 | `pass` | `pass` | 3.52541637 | 2.86810851 | -0.65730786 | 14 | 20 | 26 | 52 | -0.29687500 |

## Counts

- Gate cells: `4`
- Contextual alpha-0 loss wins: `4`
- Contextual utilization wins: `4`
- Contextual support-churn reductions: `4`
- Contextual nonzero HEP wins: `0`

## Next Step

apply the contextual MLP support-router default change in the support-wide experiment configs
