# Residual Support Width Repeat Decision

- Status: `pass`
- Decision: `satisfy_residual_support_width_repeat_gate`
- Selected direction: `residual_support_width_promotion_gate`
- Promote support-width default: `False`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`

## Rationale

The seed-2 support-width repeat has matching local and Colab artifact-backed evidence. In both backends, top-k 2 support improves ordinary alpha-0 supervised CE loss and final residual loss over the top-k 1 baseline at larger-char and tokenized scales. This satisfies the repeat gate, but it does not itself change the default support width.

## Evidence

| Backend | Scale | Artifact check | Verdict | Baseline alpha-0 | Support alpha-0 | Delta | Baseline final | Support final | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local | larger_char | `pass` | `pass` | 3.36594748 | 3.14497447 | -0.22097301 | 3.36594748 | 3.14497447 | -0.22097301 |
| local | tokenized | `pass` | `pass` | 4.14451742 | 3.59730530 | -0.54721212 | 4.14451742 | 3.59730530 | -0.54721212 |
| colab | larger_char | `pass` | `pass` | 3.36594748 | 3.20794559 | -0.15800190 | 3.36594748 | 3.20794559 | -0.15800189 |
| colab | tokenized | `pass` | `pass` | 4.14451742 | 3.58100390 | -0.56351352 | 4.14451742 | 3.58100390 | -0.56351352 |

## Next Step

define a bounded residual support-width promotion gate with exact criteria for any default support-width change
