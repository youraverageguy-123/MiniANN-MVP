# MiniANN — C++ Artificial Neural Network Library (Problem Statement 5)

A modular, lightweight feed-forward Artificial Neural Network framework written from scratch in
**modern C++17**. No PyTorch, no TensorFlow, no Eigen, no BLAS.

Goal is **architectural clarity, strict encapsulation, and clean Object-Oriented Design** —
not execution speed. You can construct layer topologies, train with analytical gradient
descent, evaluate with statistical metrics, and persist models as human-readable text.

Spec version: **1.3 / Problem Statement 5**.

## What does it do?

- Build networks as `NeuralNetwork -> Layer -> Neuron` (composition, not inheritance)
- Swap math pieces via pure-virtual interfaces:
  `IActivation` (Sigmoid / Tanh / ReLU / LeakyReLU / Swish), `ILoss` (MSE / BCE / CCE),
  `IOptimizer` (SGD / Momentum / Adam), `IMetric` (Accuracy / ConfusionMatrix),
  `ILogger` (Console), `IVisualizer` (LossCurve / AccuracyCurve / BoundaryMap /
  NetworkGraph / WeightsTable)
- Train with mini-batch backpropagation (`Trainer::fit`), Xavier/He initialization
- Load data from code (`makeAndGate()` / `makeOrGate()` / `makeXorGate()`) or CSV
  (`Dataset::loadCSV`, RFC-4180 style, header handling, train/test `split`)
- Evaluate binary and multi-class problems (threshold for 1-output, argmax otherwise,
  per-class precision/recall)
- Save/load models as plain text with 17-digit IEEE-754 precision (`MINIANN 1` format)
- Export loss curves (`CSVLossExporter`) for Excel/plotting
- Prove correctness with gradient check, round-trip, and AND/OR/XOR + Iris demos

## Repository layout

```text
MiniANN_MVP/
  CMakeLists.txt
  build.bat                  # Windows g++ build, no CMake required
  .gitignore
  README.md
  include/miniann/           # frozen interface contract
    types.hpp                # Vector, Matrix
    activation.hpp           # IActivation x5, ActivationFactory
    neuron.hpp               # Neuron, WeightInit::Uniform|Xavier|He
    layer.hpp                # Layer
    network.hpp              # NeuralNetwork
    loss.hpp                 # ILoss, MSELoss, BCELoss, CCELoss, LossFactory
    optimizer.hpp            # IOptimizer, SGD, Momentum, Adam, OptimizerFactory
    trainer.hpp              # TrainingConfig, TrainingHistory (+acc), Trainer
    dataset.hpp              # Dataset, makeAnd/Or/XorGate
    metrics.hpp              # IMetric, Accuracy, ConfusionMatrix
    logger.hpp               # LogLevel, ILogger, ConsoleLogger
    serializer.hpp           # ModelSerializer, CSVLossExporter
    exporter.hpp             # re-export of serializer
    cli.hpp                  # header-only CLI: pick optimizer/activation/training
    visualizer.hpp           # IVisualizer, LossCurve, BoundaryMap, AccuracyCurve,
                             # NetworkGraph, WeightsTable
  src/                       # one .cpp per header
    activation.cpp neuron.cpp layer.cpp network.cpp loss.cpp optimizer.cpp
    trainer.cpp dataset.cpp metrics.cpp logger.cpp serializer.cpp visualizer.cpp
  demos/
    and_or_demo.cpp          # AND + OR, 2-2-1 sigmoid, SGD
    xor_demo.cpp             # XOR with CLI: --opt/--lr/--act/--epochs/--batch/--hidden/--live
    iris_demo.cpp            # data/iris_small.csv, 4-6-3, Adam, confusion matrix
    compare_demo.cpp         # §13: same net, SGD vs Momentum vs Adam table
    playground.cpp           # §28: interactive dataset/arch/loss/opt/train/eval/predict
  tests/
    test_basic.cpp           # activations, round-trip, dim-mismatch throws
    test_choices.cpp         # LeakyReLU/Swish values + finite-diff, factory sanity
    gradient_check.cpp       # centered finite-difference vs backprop
  data/
    iris_small.csv           # 30 rows, header, 4 features + label {0,1,2}
  docs/
    architecture.md          # ownership, pipeline, seams
```

Key design rules enforced:

- `derivative(z)` takes pre-activation `z`, not output `a`
- `Neuron::backward()` accumulates (`+=`), zeroed once per batch via `zeroGradients()`
- `Layer::backward()` uses weights *before* the optimizer step
- Serializer tags `sigmoid|tanh|relu|leaky_relu|swish` match `ActivationFactory::create()` exactly
- Network never trains itself; `Trainer` + `IOptimizer` own mutation
- Flags `-std=c++17 -Wall -Wextra -Wpedantic`, no raw `new/delete`,
  `const` correctness, `PascalCase` classes / `I`-interfaces / `camelCase` methods /
  trailing `_` members

## Requirements

- Windows + PowerShell
- `g++` with C++17 (MSYS2 MinGW tested: `g++ --version` should print 13+)
- CMake is **optional** — `build.bat` uses `g++` directly
- No external libraries

## How to build

From this folder:

```bat
.\build.bat
```

Expected:

```text
Build OK
```

This builds:

| Binary | Source |
|---|---|
| `test_basic.exe` | `tests/test_basic.cpp` |
| `test_choices.exe` | `tests/test_choices.cpp` |
| `gradient_check.exe` | `tests/gradient_check.cpp` |
| `and_or_demo.exe` | `demos/and_or_demo.cpp` |
| `xor_demo.exe` | `demos/xor_demo.cpp` |
| `iris_demo.exe` | `demos/iris_demo.cpp` |
| `compare_demo.exe` | `demos/compare_demo.cpp` |
| `playground.exe` | `demos/playground.cpp` |

CMake alternative (if installed):

```bat
cmake -B build -S .
cmake --build build --config Release
```

## Choosing descent method, training setup, and activations

`xor_demo` takes CLI flags (all optional, defaults = classic `tanh` + SGD run):

```bat
.\xor_demo.exe --opt adam --lr 0.01 --act swish --epochs 3000 --batch 1 --seed 42 --hidden 4
.\xor_demo.exe --opt momentum --lr 0.1 --act leaky_relu --epochs 5000
.\xor_demo.exe --help
```

| Flag | Choices | Meaning |
|---|---|---|
| `--opt` | `sgd` (plain gradient step), `momentum` (velocity `mu=0.9` carries past updates through ravines), `adam` (per-weight adaptive rates + bias correction) | descent method, built by `OptimizerFactory::create` |
| `--lr` | float | learning rate (try `0.5` SGD/momentum, `0.01` Adam) |
| `--act` | `sigmoid` `tanh` `relu` `leaky_relu` `swish` | hidden-layer activation, built by `ActivationFactory::create` |
| `--batch` | `1` online, `>1` mini-batch, `0` full batch | training method (gradient averaging width) |
| `--epochs` `--seed` `--hidden` | ints | run length, RNG seed, hidden width |
| `--live` / `--no-live` | flag | live redraw of plots + neuron firing (default on) |

New activations:

| Function | Formula | Why it exists |
|---|---|---|
| LeakyReLU | `z>0 ? z : 0.01z` | ReLU that never fully dies: negative side keeps gradient `0.01`, fixing dead neurons |
| Swish | `z·sigmoid(z)` | Smooth, non-monotonic gate; often beats ReLU on small nets, still one line + closed-form derivative |

Both are first-class: factory tokens, `He` init auto-selected for the ReLU family,
serializer round-trips their `leaky_relu`/`swish` tags, derivatives covered by
`test_choices.exe` finite-difference checks.

## Visuals (no libraries)

Static visuals are `IVisualizer`s with `render()`, printed polymorphically:

- **Loss curve** (`LossCurve`): true `*` polyline with y-axis tick labels and an
  epoch x-axis (`1 → N`). Log-y kicks in when the span exceeds 50x, so one plot
  shows both the early plunge and the late fine convergence.
- **Accuracy curve** (`AccuracyCurve`): same line plot on the fixed 0..1 scale.
- **Decision boundary** (`BoundaryMap`): 41×21 map of the unit square,
  `#` = predicts 1, `.` = predicts 0, with training points stamped as `0`/`1`
  so misclassifications are visible at a glance.
- **Network structure** (`NetworkGraph`): one line per layer — input dim, neuron count,
  activation — e.g. `in(2) ==> [8 x tanh] ==> [8 x tanh] ==> [1 x sigmoid] => out`.
- **Learned weights** (`WeightsTable`): one row per neuron, `w=[...] b=...`.

### Live training view

`xor_demo` (and the playground) redraw the screen as epochs stream in via the
`TrainingCallback` hook — loss line plot, accuracy line plot, and a **neuron panel**
showing every neuron firing on a probe input (`-` = off, `#` = firing, `|` = center):

```bat
.\xor_demo.exe --epochs 2000            :: live view (default)
.\xor_demo.exe --no-live --epochs 2000  :: plain log, for pipes/files
```

Needs an ANSI-capable terminal (Windows Terminal / PowerShell); use `--no-live`
otherwise.

### Browser reports

Every run also writes a self-contained HTML report with real SVG line charts
(loss log-y + accuracy), an SVG decision heatmap, and the network/weights —
no dependencies, just open the file:

- `xor_demo` → `xor_report.html`
- `compare_demo` → `compare_report.html` (all three optimizers overlaid)
- `playground` → `playground_report.html`

## Loss functions

`ILoss` has three implementations behind `LossFactory::create("mse|bce|cce")`:

| Loss | Use with | Notes |
|---|---|---|
| MSE | any head | mean squared error, the default |
| BCE | 1-output sigmoid | binary cross-entropy, predictions clipped to `[1e-12, 1-1e-12]`; rejects multi-output |
| CCE | one-hot multi-output | categorical cross-entropy `-mean(t·log p)`; pairs with sigmoid heads as in `iris_demo` |

Gradients are covered by centered finite-difference asserts in `test_choices.exe`.
`playground` lets you pick the loss at the prompt; `xor_demo` uses MSE.

## Optimizer comparison

```bat
.\compare_demo.exe
```

Trains the identical network (`2 → 8 → 8 → 1`, tanh/tanh/sigmoid, same seed/init)
with SGD, Momentum, and Adam on XOR and prints one table from real runs:

```text
| Optimizer | LR   | Epochs | Final Loss | Accuracy |
| sgd       | 0.5  | 3000   | 1.2568e-05 | 100.0%   |
| momentum  | 0.1  | 3000   | 1.2965e-05 | 100.0%   |
| adam      | 0.01 | 3000   | 2.2844e-07 | 100.0%   |
```

followed by each optimizer's loss curve, so convergence speed is visible, not claimed.

## Interactive playground

```bat
.\playground.exe
```

No C++ edits needed to experiment — the full revised-spec loop:

```text
MiniANN Playground
Dataset: 1=AND 2=OR 3=XOR 4=Iris 5=CSV file   (path, header, target col/dim,
                                               optional one-hot + max-abs norm)
  -> train/test split
  -> architecture, e.g. "2 8 8 1" + "tanh tanh sigmoid"
  -> loss mse|bce|cce, optimizer sgd|momentum|adam, lr, epochs, batch, seed
  -> train (network graph first, live log)
  -> loss curve, accuracy curve, boundary map (2-D), weights table,
     test accuracy, confusion matrix, saved model + loss CSV
  -> prediction REPL: type features, get class + confidence, q to quit
```

Example session: pick `3` (XOR), sizes `2 4 1`, activations `tanh sigmoid`,
loss `mse`, optimizer `sgd` — then type `1 0` at the `>` prompt.

## How to test (and what pass looks like)

Run in order:

```bat
.\test_basic.exe
.\test_choices.exe
.\gradient_check.exe
.\and_or_demo.exe
.\xor_demo.exe
.\iris_demo.exe
.\compare_demo.exe
.\playground.exe
```

### 1. `test_basic.exe` — smoke test

Checks `sigmoid(0)=0.5`, `relu(-2)=0`, save/load round-trip `<1e-14`,
dimension mismatch throws `invalid_argument`.

Pass looks like:

```text
roundtrip diff=0
ALL MVP CHECKS PASS
```

### 2. `gradient_check.exe` — backprop correctness

Compares analytical gradients vs centered difference
`g_num = (L(t+e)-L(t-e))/2e`, `e=1e-5`. Required relative error `<1e-6`.

Pass looks like:

```text
params=13 rel_error=3.40052e-11
GRADIENT CHECK PASS
```

If this fails, bug is in `src/neuron.cpp`, `src/layer.cpp`, or `src/network.cpp`.

### 3. `and_or_demo.exe` — linearly separable gates

Trains `2-2-1` sigmoid, SGD `lr=0.5`, 3000 epochs. Pass = loss drops
`0.20 -> ~0.0002` and outputs split cleanly:

```text
=== AND final_loss=0.000255087 ===
 [0,0] -> 0.0004
 [0,1] -> 0.0156
 [1,0] -> 0.0124
 [1,1] -> 0.9751
=== OR final_loss=0.0001 ===
 [0,0] -> 0.0161
 [0,1] -> 0.9884
 ...
```

### 4. `xor_demo.exe` — non-linear proof

Trains `2-4-1` tanh/sigmoid, SGD `lr=0.5`, 10000 epochs. Also writes
`xor_loss.csv` and `xor.model`, then reloads and compares.

Pass looks like:

```text
[INFO] epoch 10000/10000 train_loss=2.56533e-05
XOR results (2-4-1 tanh/sigmoid, SGD lr=0.5):
 [0,0] -> 0.0023 (target 0.0000)
 [0,1] -> 0.9946 (target 1.0000)
 [1,0] -> 0.9946 (target 1.0000)
 [1,1] -> 0.0062 (target 0.0000)
accuracy=100% final_loss=0.0000
serialization round-trip max|y_orig - y_loaded|=0.0000
```

### 5. `iris_demo.exe` — CSV + multi-class

Loads `data/iris_small.csv`, normalizes, 80/20 `split()`, one-hot encodes,
trains `4-6-3` tanh/sigmoid with Adam `lr=0.01`, batch 8. Writes
`iris_loss.csv`, `iris.model`.

Pass looks like:

```text
[INFO] epoch 1500/1500 train_loss=0.000191913 val_loss=0.000116322
train acc=1
test  acc=1
confusion (rows=true, cols=pred):
2 0 0
0 1 0
0 0 3
class 0 prec=1 rec=1
class 1 prec=1 rec=1
class 2 prec=1 rec=1
```

## How to use it in your own program

Minimal XOR example:

```cpp
#include "miniann/network.hpp"
#include "miniann/dataset.hpp"
#include "miniann/loss.hpp"
#include "miniann/optimizer.hpp"
#include "miniann/trainer.hpp"
#include "miniann/logger.hpp"
#include "miniann/serializer.hpp"
#include <random>

using namespace miniann;

int main() {
    std::mt19937 rng(42);
    Dataset data = makeXorGate();

    NeuralNetwork net;
    net.addLayer(Layer(4, 2, ActivationFactory::create("tanh"), rng, WeightInit::Xavier));
    net.addLayer(Layer(1, 4, ActivationFactory::create("sigmoid"), rng, WeightInit::Xavier));

    MSELoss loss;
    SGD opt(0.5);
    ConsoleLogger logger;
    Trainer trainer(net, loss, opt, &logger);

    TrainingConfig cfg;
    cfg.epochs = 10000;
    cfg.batchSize = 1;   // 1 = online SGD, >1 = mini-batch, 0 = full batch
    cfg.shuffle = true;
    cfg.seed = 42;
    cfg.logEvery = 2000;

    TrainingHistory h = trainer.fit(data, nullptr, cfg);
    CSVLossExporter::exportHistory(h, "my_loss.csv");
    ModelSerializer::save(net, "my.model");

    Vector p = net.predict({1.0, 0.0}); // ~1.0
}
```

Compile it:

```bat
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src\activation.cpp src\neuron.cpp src\layer.cpp src\network.cpp src\loss.cpp src\optimizer.cpp src\trainer.cpp src\dataset.cpp src\metrics.cpp src\logger.cpp src\serializer.cpp my_file.cpp -o my_app.exe
```

CSV usage:

```cpp
Dataset raw = Dataset::loadCSV("data/iris_small.csv", true, 4, 1);
// hasHeader=true, target column index 4, target dim 1
// inputs = all other columns, targets = columns [4, 5)
raw.validate();
auto [train, test] = raw.split(0.8, rng);
```

Metrics usage:

```cpp
Accuracy acc;
double a = acc.evaluate(preds, targets);
ConfusionMatrix cm(3);
double acc2 = cm.evaluate(preds, targets);
cm.print();
cm.precision(0); cm.recall(0);
```

Model file format (`my.model`):

```text
MINIANN 1
layers 2
layer 4 2 tanh
<4 rows of: w0 w1 bias>
layer 1 4 sigmoid
<1 row of: w0 w1 w2 w3 bias>
```

All floats use `setprecision(17)`. Load validates magic, version, counts,
`numInputs+1` values per row, throws `runtime_error` on malformed files.

Loss curve (`my_loss.csv`):

```text
epoch,train_loss,val_loss,train_acc,val_acc
1,0.25178,,0.5,
...
```

## Pushing to GitHub

`.gitignore` already excludes build noise:

- `*.exe`, `*.o`, `*.obj`, `build/`, CMake cache
- `*.model`, root `*_loss.csv` (regenerated on every run)
- `.vs/`, `.vscode/`, `Thumbs.db`, `.DS_Store`

Tracked: `include/`, `src/`, `demos/`, `tests/`, `data/*.csv`,
`docs/`, `CMakeLists.txt`, `build.bat`, `README.md`.

Do not init git in the parent `Documents` folder. Init here:

```bat
cd MiniANN_MVP
git init
git add .gitignore CMakeLists.txt build.bat README.md include src demos tests data docs
git status
git commit -m "MiniANN MVP v1.3"
gh repo create miniann-mvp --private --source=. --push
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `.\build.bat` not recognized | You are not in `MiniANN_MVP`; `cd` there first, use `.\build.bat` with the dot-backslash |
| `g++ not recognized` | Install MSYS2 MinGW and add to PATH, restart PowerShell |
| `cannot open datasets/circles.csv` | Old prototype path; MVP uses `data/iris_small.csv`, run from `MiniANN_MVP` |
| XOR stuck at 75% | Known local minimum for `2-2-1` sigmoid + Adam; use `2-4-1` tanh + SGD as in `demos/xor_demo.cpp` |
| CSV load throws ragged / non-numeric | Check header flag, commas, no blank mixed-width rows; see `data/iris_small.csv` for format |

## Out of scope (by design)

CNN/RNN, GPU, autograd, Dropout, optimizer-state serialization. See `docs/architecture.md`
for ownership, pipeline, and integration seams.
