# Report-2 Prep — MiniANN as a Library (how it works + OOP map)

Read this top-to-bottom before the meeting. Every claim has a file you can open.

## 1. One-paragraph story
MiniANN is a small **installable C++17 library** (`miniann::miniann` target, one
`#include "miniann/miniann.hpp"`) with demos on top. Core owns math only;
demos/GUIs own choices. See `examples/minimal_xor.cpp` (~30 lines: dataset →
network → loss/opt → `Trainer::fit` → `Accuracy` → `ModelSerializer::save`).

## 2. Data flow (the 6 steps)
1. **Data** — `include/miniann/dataset.hpp`, `src/dataset.cpp`: `loadCSV`
   (RFC-4180, `-1` = last col, ragged/non-numeric throws), `validate()`,
   `split(frac, rng)`; helpers in `include/miniann/preprocessing.hpp`
   (`normalizeMaxAbs`, `oneHotEncode`, `detectCsvColumnCount`).
2. **Build** — `network.hpp/layer.hpp/neuron.hpp`: `NeuralNetwork ♦→ Layer ♦→
   Neuron`; `ActivationFactory::create("tanh|…|linear|softmax")`,
   `cli::defaultInitFor` picks He/Xavier.
3. **Train** — `trainer.hpp/cpp`: shuffle (`cfg.seed`), batch (`1`=online,
   `0`=full), per-batch `zeroGradients()`, per-sample
   `predict → loss.gradient → backward(+=) → optimizer.step(÷B)`,
   `TrainingCallback::onEpoch` notifies UI.
4. **Optimize** — `optimizer.hpp/cpp`: `SGD / Momentum(mu=0.9) / Adam`
   via `OptimizerFactory`; Adam/Momentum expose `state()/restore()`.
5. **Evaluate** — `metrics.hpp/cpp`: `Accuracy(threshold)`,
   `ConfusionMatrix(n, threshold)` + `precision/recall/f1/macroF1/toString`.
6. **Persist/visualize** — `serializer.hpp/cpp`: `MINIANN 2`
   (`layer n m act[:alpha]`, trailing `opt` block; v1 still loads),
   `CSVLossExporter`; `visualizer.hpp` + `report.hpp`:
   `LossCurve/AccuracyCurve/BoundaryMap/NetworkGraph/WeightsTable`,
   `HtmlReport + IHtmlSection/HtmlReportBuilder`.

## 3. OOP map (say file + line in viva)
| Pillar | Where | Line to quote |
|---|---|---|
| Abstraction | `IActivation/ILoss/IOptimizer/IMetric/ILogger/IVisualizer/TrainingCallback` | `activation.hpp:10`, `loss.hpp:9`, `optimizer.hpp:10` |
| Inheritance | `Tanh: IActivation`, `Adam: IOptimizer`, `LossCurve: IVisualizer`, `ConfigSection: IHtmlSection` | `activation.hpp:30`, `report.hpp:ConfigSection` |
| Polymorphism | `vector<unique_ptr<IVisualizer>> → v->render()`; factories `create(name)` | `demos/xor_demo.cpp` visuals loop |
| Encapsulation | net never trains itself; `Trainer+IOptimizer` mutate | `network.hpp` has no `fit()` |
| Composition | `Network ♦→ Layer ♦→ Neuron`; report holds sections | `network.hpp:18`, `visualizer.hpp:HtmlReport` |
| SRP | `JsModelExporter` reads getters only; `preprocessing.hpp` pure functions | `visualizer.hpp:JsModelExporter` |
| Open/Closed | new `IHtmlSection` without editing `HtmlReport::render` | `report.hpp` |
| Strategy | `IOptimizer`, `ILoss` swapped by string | `cli.hpp:makeOptimizer` |
| Observer | `Trainer::fit(..., cb)` → console/Qt/HTML | `trainer.hpp:TrainingCallback` |
| Factory | `Activation/Optimizer/LossFactory` | `src/activation.cpp:5` |
| Builder | `HtmlReportBuilder().withSeries()…build()` | `report.hpp` |

Softmax note (they will ask): scalar `IActivation` can't express a vector op,
so `Softmax::isVector()==true`, `Layer::forward` does stable joint softmax and
`backward` applies the Jacobian (`src/layer.cpp`). Per-neuron `activate()`
throws by design.

## 4. What's new since Report-1 (small, modular)
- `linear|softmax` + LeakyReLU `alpha` validation/persist (`activation.hpp/cpp`, `layer.cpp`, `neuron.cpp`).
- `Layer fanOut` param (correct Xavier; default = old behavior).
- `Accuracy(threshold)`, `CM::f1/macroF1/toString` (`metrics.hpp/cpp`).
- `MINIANN 2` + `save(net, opt)` / `loadWithOptimizer` (`serializer.hpp/cpp`).
- Library cut: `miniann.hpp` umbrella, `miniann::miniann` alias,
  `target_compile_features cxx_std_17`, GNUInstallDirs install + export,
  `examples/minimal_xor.cpp`; shared `preprocessing.hpp` dedupes Iris/playground/GUI copies.
- Qt without collisions: `gui_contract.hpp` (`WorkbenchConfig`), friend's
  `demos/gui_qt.cpp` + `experiment.hpp/cpp`; `docs/qt_plan.md` task split.

## 5. Merge-conflict rules (state these first)
- Friend: `demos/gui_qt.cpp`, `include/miniann/experiment.hpp`,
  `src/experiment.cpp`, `tests/test_training.cpp` only.
- Me: core `src/*`, `include/miniann/*` (except qt_*), `playground/cli/report`,
  `examples/*`, `docs/*`.
- Shared touchpoints: `gui_contract.hpp` + model file schema only.
- Never commit: `*.exe/*.model/*.csv/*_report.html`, `build/`, local notes
  (`MY_NOTES*`, `*SUMMARY*` — already gitignored).

## 6. Likely questions + 1-line answers
- *Why not Eigen?* Clarity over speed; explicit loops show backprop ownership.
- *Who mutates weights?* Only `IOptimizer::step` via `Neuron::applyStep`.
- *Why `derivative(z)`?* Needs pre-activation; cached `lastZ_` per neuron.
- *Why `+=` grads?* Mini-batch accumulation, zeroed once per batch.
- *Resume training?* `save(net, opt)` + `loadWithOptimizer` restores Adam `m/v/t`.
- *Threshold?* `Accuracy(0.9)` demo in `tests/test_correctness.cpp:6`.
- *Install as lib?* `cmake --install build` → `find_package(MiniANN)` →
  `target_link_libraries(app miniann::miniann)`.
