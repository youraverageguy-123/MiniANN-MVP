@echo off
REM ========================================================
REM Build MiniANN MVP & Raylib GUI Demo
REM ========================================================

set INCL=-Iinclude
set SRC=src\activation.cpp src\neuron.cpp src\layer.cpp src\network.cpp src\loss.cpp src\optimizer.cpp src\trainer.cpp src\dataset.cpp src\metrics.cpp src\logger.cpp src\serializer.cpp src\visualizer.cpp
set FLAGS=-std=c++17 -O2 -Wall -Wextra -Wpedantic
set RAYLIB_LIBS=-lraylib -lopengl32 -lgdi32 -lwinmm

echo [1/9] Building xor_demo.exe...
g++ %FLAGS% %INCL% %SRC% demos\xor_demo.cpp -o xor_demo.exe
if errorlevel 1 exit /b 1

echo [2/9] Building and_or_demo.exe...
g++ %FLAGS% %INCL% %SRC% demos\and_or_demo.cpp -o and_or_demo.exe
if errorlevel 1 exit /b 1

echo [3/9] Building iris_demo.exe...
g++ %FLAGS% %INCL% %SRC% demos\iris_demo.cpp -o iris_demo.exe
if errorlevel 1 exit /b 1

echo [4/9] Building compare_demo.exe...
g++ %FLAGS% %INCL% %SRC% demos\compare_demo.cpp -o compare_demo.exe
if errorlevel 1 exit /b 1

echo [5/9] Building playground.exe...
g++ %FLAGS% %INCL% %SRC% demos\playground.cpp -o playground.exe
if errorlevel 1 exit /b 1

echo [6/9] Building gradient_check.exe...
g++ %FLAGS% %INCL% %SRC% tests\gradient_check.cpp -o gradient_check.exe
if errorlevel 1 exit /b 1

echo [7/9] Building test_basic.exe...
g++ %FLAGS% %INCL% %SRC% tests\test_basic.cpp -o test_basic.exe
if errorlevel 1 exit /b 1

echo [8/9] Building test_choices.exe...
g++ %FLAGS% %INCL% %SRC% tests\test_choices.cpp -o test_choices.exe
if errorlevel 1 exit /b 1

echo [9/9] Building gui_app.exe (Raylib)...
g++ %FLAGS% %INCL% %SRC% demos\gui_app.cpp %RAYLIB_LIBS% -o gui_app.exe
if errorlevel 1 (
    echo [ERROR] Raylib GUI build failed!
    exit /b 1
)

echo.
echo ====================================
echo  All targets built successfully!
echo ====================================
echo Launching GUI app...
.\gui_app.exe