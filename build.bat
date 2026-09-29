@echo off
REM Build MiniANN MVP without CMake (g++ directly). Run from MiniANN_MVP dir.
set INCL=-Iinclude
set SRC=src\activation.cpp src\neuron.cpp src\layer.cpp src\network.cpp src\loss.cpp src\optimizer.cpp src\trainer.cpp src\dataset.cpp src\metrics.cpp src\logger.cpp src\serializer.cpp
set FLAGS=-std=c++17 -O2 -Wall -Wextra -Wpedantic
g++ %FLAGS% %INCL% %SRC% demos\xor_demo.cpp -o xor_demo.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% demos\and_or_demo.cpp -o and_or_demo.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% demos\iris_demo.cpp -o iris_demo.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% tests\gradient_check.cpp -o gradient_check.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% tests\test_basic.cpp -o test_basic.exe
if errorlevel 1 exit /b 1
echo Build OK
