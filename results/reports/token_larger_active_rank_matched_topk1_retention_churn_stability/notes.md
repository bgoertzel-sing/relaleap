# Active Top-k-1 Retention/Churn Stability

- Status: `pass`
- Decision: `active_topk1_retention_churn_stable_across_local_seeds`
- Packet count: `2`
- Mean top-k-1 support churn: `0.005859375`
- Mean top-k-2 support churn: `0.86328125`
- Minimum support-churn advantage: `0.8046875`
- Minimum logit-churn advantage: `0.011922299861907959`
- Minimum transfer-improvement advantage: `0.01635599136352539`

## Rationale

Both local seed packets pass the active top-k-1 retention/churn probe. Rank-matched contextual top-k-1 has much lower support churn than the promoted top-k-2 reference in each packet, no higher logit churn, and at least comparable transfer CE improvement. This establishes a local retention/churn stability bracket, while preserving the negative singleton-gain caveat from the source separability packet.

## Caveat

The source separability packet still reports negative top-k-1 singleton gain, so this summary should be read as retention/churn stability evidence for the active bracket, not as a singleton causal separability claim.

## Next Step

decide whether the local retention/churn stability bracket warrants a targeted Colab/GPU replication or should remain local support for the active rank-matched top-k-1 causal bracket
