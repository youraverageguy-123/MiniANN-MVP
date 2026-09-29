# MiniANN Architecture Note (Problem Statement 5, v1.3)

## Ownership
`NeuralNetwork ♦→ Layer ♦→ Neuron`. Network owns forward/backward chaining only.
`Trainer` + `IOptimizer` own training policy and parameter mutation.
`Dataset` owns data. `ILoss` / `IActivation` / `IMetric` / `ILogger` are pure interfaces.

## Pipeline
1. `Dataset::validate()` (dims, NaN/Inf), optional `split()`.
2. `Trainer::fit()`: shuffle (mt19937, `cfg.seed`), batch partition
   (`batchSize==0` = full batch), per-batch `zeroGradients()`.
3. Per sample: `predict()` caches `lastInputs_`/`lastZ_` → `loss.gradient()` →
   `backward()` accumulates `gradW += δ·a`, `gradB += δ`.
4. `optimizer.step(net, batchSize)`: SGD averages by `1/B`; Adam tracks
   bias-corrected `m̂,v̂` per weight.
5. Log via `ILogger`, export via `CSVLossExporter`, persist via `ModelSerializer`.

## Critical seams
- `derivative(z)` uses pre-activation `z`.
- Gradients accumulate (`+=`), zeroed once per batch.
- `Layer::backward` reads weights before optimizer step.
- Serializer tokens `sigmoid|tanh|relu` match `ActivationFactory::create`.

## Verification
`test_basic` (activations, round-trip <1e-14, dim throws),
`gradient_check` (centered diff ε=1e-5, rel-err <1e-6),
`and_or_demo` / `xor_demo` (100%), `iris_demo` (CSV + confusion matrix).
