# Qt Workbench Plan (friend owns Qt GUI, I own core/library)

## Where things live (friend PUSHED — real file wins)
- Friend: `demos/gui_qt.cpp` (+ `include/miniann/experiment.hpp`,
  `src/experiment.cpp`, `tests/test_training.cpp`). Only he touches these.
- Shared: `gui_contract.hpp`, core lib, `cli`, `report`, `serializer`
  (MINIANN 2 + optimizer resume), `preprocessing.hpp`, `miniann.hpp`.

## Build (all optional, core never breaks)
- CMake: `find_package(Qt6 Widgets Charts)` — `qt_app` builds only if found.
- `build.bat`: new step `[11/11] qt_app.exe (Qt6, optional)` — warns + continues if Qt missing.
- MSYS2: `pacman -S mingw-w64-x86_64-qt6-base mingw-w64-x86_64-qt6-charts`
- PowerShell: set `CMAKE_PREFIX_PATH=C:\msys64\mingw64` then `cmake -B build && cmake --build build`.

## Feature parity checklist (both UIs)
1. Datasets: AND/OR/XOR/Iris/CSV (+ drag-drop in Qt via file dialog).
2. Architecture editor: H1/H2 steppers, topology label `[in->h1->h2->out]`.
3. Activations H1/H2: sigmoid|tanh|relu|leaky_relu|swish + output: sigmoid|softmax|linear.
4. Loss/optimizer: mse|bce|cce + sgd|momentum|adam, lr box, epochs slider, batch box.
5. Live curves: loss + accuracy tabs, train vs val, hover tooltip, RUN/IDLE badge.
6. Threshold slider (binary) wired to `Accuracy(threshold)` + `ConfusionMatrix::toString/f1/macroF1`.
7. Save/load: `ModelSerializer::save(net, opt, path)` + `loadWithOptimizer` (resume training).
8. Export: `HtmlReportBuilder` one-click `*_report.html` + `CSVLossExporter`.

## My next core tasks (unblocks Qt)
- [x] softmax/linear joint layer, LeakyReLU alpha persist, optimizer resume
- [x] `Accuracy(threshold)`, `ConfusionMatrix::f1/macroF1/toString`
- [ ] LR schedule + early stopping hooks (next, both UIs get it free via Trainer)

## Friend's next Qt tasks (your files only)
- [ ] `TrainWorker : QObject` in thread, `TrainingCallback` → signals (`epochDone`, `finished`)
- [ ] QtCharts loss/acc tabs + threshold `QSlider` + confusion `QTableWidget`
- [ ] File menu: open CSV, save/load `.model` (with optimizer), export HTML/CSV
- [ ] Status bar: epoch/loss/acc + error toast on `std::exception`
