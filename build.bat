@echo off
REM ========================================================
REM Build MiniANN MVP and Qt Widgets GUI (two-stage: lib once, link each)
REM TOOLCHAIN: build with an x86_64 UCRT g++ (MSYS2 ucrt64 or WinLibs-POSIX-UCRT).
REM MSVCRT-built exes + UCRT Qt DLLs segfault in <fstream> (Iris preset died
REM loading its CSV). If plain `g++` is MSVCRT, prepend the UCRT bin dir first:
REM   set PATH=C:\...\WinLibs.POSIX.UCRT\mingw64\bin;%PATH%
REM Qt6 note: needs MSYS2 mingw-w64-ucrt-x86_64-qt6-base installed.
REM No moc step required: demos\gui_qt.cpp uses no Q_OBJECT.
REM ========================================================

REM Auto-select a UCRT g++ when one is installed (Qt6 DLLs are UCRT builds;
REM an MSVCRT-built gui_qt.exe segfaults in <fstream> while loading CSVs,
REM e.g. the Iris dataset. A UCRT g++ already first on PATH is left alone.
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*") do (
    if exist "%%~D\mingw64\bin\g++.exe" set "PATH=%%~D\mingw64\bin;%PATH%"
)
g++ --version | findstr /i "ucrt" >nul && echo [toolchain] UCRT g++ selected || echo [toolchain] WARNING: no UCRT g++ found, using default g++ (Qt GUI may be unstable)

set INCL=-Iinclude
set FLAGS=-std=c++17 -O2 -Wall -Wextra -Wpedantic -static-libstdc++ -static-libgcc -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic
REM Static runtime: exes never depend on PATH-ordered MinGW DLLs (a second Qt
REM copy on PATH, e.g. MiKTeX, plus chained MinGW runtimes segfaulted fstream).
set QTINC=-IC:/msys64/ucrt64/include/qt6 -IC:/msys64/ucrt64/include/qt6/QtWidgets -IC:/msys64/ucrt64/include/qt6/QtGui -IC:/msys64/ucrt64/include/qt6/QtCore -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB
set QTLIBS=-mwindows -LC:/msys64/ucrt64/lib -lQt6Widgets -lQt6Gui -lQt6Core

if not exist obj mkdir obj

echo Compiling MiniANN library (once, reused by all targets)...
for %%f in (src\activation.cpp src\neuron.cpp src\layer.cpp src\network.cpp src\loss.cpp src\optimizer.cpp src\trainer.cpp src\dataset.cpp src\metrics.cpp src\logger.cpp src\serializer.cpp src\visualizer.cpp src\experiment.cpp) do (
    g++ %FLAGS% %INCL% -c %%f -o obj\%%~nf.o
    if errorlevel 1 exit /b 1
)
set LIB=obj\*.o

if /i "%1"=="gui" goto build_gui
if /i "%1"=="clean" (
    call clean.bat
    exit /b 0
)
if /i "%1"=="demos" goto build_demos
if /i "%1"=="tests" goto build_tests

echo [1/11] Building xor_demo.exe...
g++ %FLAGS% %INCL% %LIB% demos\xor_demo.cpp -o xor_demo.exe
if errorlevel 1 exit /b 1

echo [2/11] Building and_or_demo.exe...
g++ %FLAGS% %INCL% %LIB% demos\and_or_demo.cpp -o and_or_demo.exe
if errorlevel 1 exit /b 1

echo [3/11] Building iris_demo.exe...
g++ %FLAGS% %INCL% %LIB% demos\iris_demo.cpp -o iris_demo.exe
if errorlevel 1 exit /b 1

echo [4/11] Building compare_demo.exe...
g++ %FLAGS% %INCL% %LIB% demos\compare_demo.cpp -o compare_demo.exe
if errorlevel 1 exit /b 1

echo [5/11] Building playground.exe...
g++ %FLAGS% %INCL% %LIB% demos\playground.cpp -o playground.exe
if errorlevel 1 exit /b 1

echo [6/11] Building gradient_check.exe...
g++ %FLAGS% %INCL% %LIB% tests\gradient_check.cpp -o gradient_check.exe
if errorlevel 1 exit /b 1

echo [7/11] Building test_basic.exe...
g++ %FLAGS% %INCL% %LIB% tests\test_basic.cpp -o test_basic.exe
if errorlevel 1 exit /b 1

echo [8/11] Building test_choices.exe...
g++ %FLAGS% %INCL% %LIB% tests\test_choices.cpp -o test_choices.exe
if errorlevel 1 exit /b 1

echo [9/11] Building test_training.exe...
g++ %FLAGS% %INCL% %LIB% tests\test_training.cpp -o test_training.exe
if errorlevel 1 exit /b 1

echo [10/11] Building test_correctness.exe...
g++ %FLAGS% %INCL% %LIB% tests\test_correctness.cpp -o test_correctness.exe
if errorlevel 1 exit /b 1

goto done

:build_gui
echo [GUI] Building gui_qt.exe (Qt Widgets)...
g++ %FLAGS% %INCL% %QTINC% %LIB% demos\gui_qt.cpp %QTLIBS% -o gui_qt.exe
if errorlevel 1 (
    echo [WARN] gui_qt.exe failed to build. Ensure window is closed and Qt6 is installed.
    exit /b 1
)
echo [GUI] gui_qt.exe built successfully!
exit /b 0

:done
echo.
echo ====================================
echo  All targets built successfully!
echo ====================================
echo Run gui_qt.exe to launch the clickable GUI (needs Qt6).
echo NOTE: after building, run deploy_qt.ps1 once so gui_qt.exe works by
echo double-click even if another Qt (e.g. MiKTeX) is on PATH:
echo   powershell -ExecutionPolicy Bypass -File .\deploy_qt.ps1
