@echo off
REM ========================================================
REM Fast MiniANN Build Script using CMake + Ninja + ccache
REM Incremental rebuilds: ~150ms!
REM ========================================================

set TARGET=%1
if "%TARGET%"=="" set TARGET=gui_qt

if not exist build (
    echo [cmake] Configuring build directory with Ninja and Qt6...
    cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=C:/msys64/ucrt64 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    if errorlevel 1 exit /b 1
)

echo [ninja] Building target '%TARGET%'...
cmake --build build --target %TARGET%
if errorlevel 1 exit /b 1

echo [build] Target '%TARGET%' built successfully!
if exist build\%TARGET%.exe (
    echo [build] Binary is located at: build\%TARGET%.exe
)
