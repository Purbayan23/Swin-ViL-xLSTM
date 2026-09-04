# A1-N Numerical-Safety Intervention

## Scope

A1 remains frozen. A1-N is a separate diagnostic experiment namespace for the
original bidirectional A1 model. It changes only how the existing mLSTM
normalizer is evaluated; it does not change the model's layers, gates,
parameters, initialization, data protocol, loss, optimizer, scheduler, or
training hyperparameters. It is not a proposed final architecture.

## Numerical change

The original normalization is mathematically equivalent to

```text
normalizer = max(abs(combination.sum(...)), exp(-max_log_decay))
output = (combination / (normalizer + eps)) @ values
```

A1-N avoids materializing `exp(-max_log_decay)` for the comparison. It computes

```text
log_abs_combination = log(abs(combination.sum(...)))
log_exp_term = -max_log_decay
log_normalizer = max(log_abs_combination, log_exp_term)
inverse_normalizer = exp(-log_normalizer)
normalized_combination =
    (combination * inverse_normalizer) / (1 + eps * inverse_normalizer)
```

This preserves the existing epsilon behavior algebraically while evaluating
the normalizer in the log domain. A1-N does not compute `exp(log_exp_term)` for
comparison, and it does not clamp values, change gate equations, or add any
other numerical mitigation.

The existing forensic observer records `max_log_decay`, cumulative log decay,
the normalized combination, `log_normalizer`, and `inverse_normalizer` in place
of the overflow-prone diagnostic `exp(-max_log_decay)`.

## Validation status

- Focused A1-N numerical tests pass, including safe-range equivalence, the
  FP32 overflow case for the original exponential, zero/small combination
  sums, and safe-input A1 inference equivalence.
- The complete lightweight test suite passes.
- A bounded one-batch CPU A1-N sanity run passes with finite forward values,
  loss, gradients, optimizer state, scheduler step, metrics, and checkpoint
  output.
- No 100-epoch experiment was run here. A1-N remains diagnostic evidence only;
  it does not replace any A1 historical result.
