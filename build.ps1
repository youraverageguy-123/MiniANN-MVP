# ========================================================
# MiniANN High-Speed Parallel & Incremental Build Script
# ========================================================
param (
    [string]$Target = "all",
    [switch]$Clean,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    Write-Host "[clean] Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path "obj") { Remove-Item -Recurse -Force "obj" }
    Get-ChildItem -Filter "*.exe" | Remove-Item -Force
    Get-ChildItem -Filter "*.o" -Recurse | Remove-Item -Force
    Write-Host "[clean] Done." -ForegroundColor Green
    exit 0
}

# Auto-detect UCRT g++
$wingetDir = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($wingetDir -and (Test-Path "$($wingetDir.FullName)\mingw64\bin\g++.exe")) {
    $env:PATH = "$($wingetDir.FullName)\mingw64\bin;$env:PATH"
}

$gppVer = (& g++ --version 2>&1 | Out-String)
if ($gppVer -match "ucrt") {
    Write-Host "[toolchain] UCRT g++ selected" -ForegroundColor Cyan
} else {
    Write-Host "[toolchain] WARNING: Default g++ selected (Qt GUI may require UCRT runtime)" -ForegroundColor Yellow
}

$INCL = "-Iinclude"
$FLAGS = @("-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-static-libstdc++", "-static-libgcc", "-Wl,-Bstatic", "-lwinpthread", "-Wl,-Bdynamic")
$QTINC = @("-IC:/msys64/ucrt64/include/qt6", "-IC:/msys64/ucrt64/include/qt6/QtWidgets", "-IC:/msys64/ucrt64/include/qt6/QtGui", "-IC:/msys64/ucrt64/include/qt6/QtCore", "-DQT_WIDGETS_LIB", "-DQT_GUI_LIB", "-DQT_CORE_LIB")
$QTLIBS = @("-mwindows", "-LC:/msys64/ucrt64/lib", "-lQt6Widgets", "-lQt6Gui", "-lQt6Core")

if (-not (Test-Path "obj")) { New-Item -ItemType Directory "obj" | Out-Null }

$srcFiles = Get-ChildItem "src/*.cpp" | ForEach-Object { "src/$($_.Name)" }


# Incremental compilation check
$toCompile = @()
foreach ($src in $srcFiles) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($src)
    $obj = "obj/$base.o"
    if ($Force -or (-not (Test-Path $obj)) -or ((Get-Item $src).LastWriteTime -gt (Get-Item $obj).LastWriteTime)) {
        $toCompile += $src
    }
}

if ($toCompile.Count -gt 0) {
    Write-Host "[build] Compiling $($toCompile.Count) modified source file(s) into obj/..." -ForegroundColor Cyan
    foreach ($src in $toCompile) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($src)
        $obj = "obj/$base.o"
        Write-Host "  -> Compiling $src" -ForegroundColor Gray
        & g++ $FLAGS $INCL -c $src -o $obj
        if ($LASTEXITCODE -ne 0) { throw "Failed to compile $src" }
    }
} else {
    Write-Host "[build] Library obj/ files are up to date (incremental cache hit)." -ForegroundColor Green
}

$LIB = Get-ChildItem "obj/*.o" | ForEach-Object { $_.FullName }

function Build-Target($name, $srcPath) {
    $exe = "$name.exe"
    $needsBuild = $Force -or (-not (Test-Path $exe)) -or ((Get-Item $srcPath).LastWriteTime -gt (Get-Item $exe).LastWriteTime)
    if (-not $needsBuild) {
        foreach ($o in $LIB) {
            if ((Get-Item $o).LastWriteTime -gt (Get-Item $exe).LastWriteTime) { $needsBuild = $true; break }
        }
    }
    if ($needsBuild) {
        Write-Host "[build] Linking $exe..." -ForegroundColor Cyan
        & g++ $FLAGS $INCL $LIB $srcPath -o $exe
        if ($LASTEXITCODE -ne 0) { throw "Failed to link $exe" }
        Write-Host "[build] OK: $exe" -ForegroundColor Green
    } else {
        Write-Host "[build] Target $exe is up to date." -ForegroundColor Gray
    }
}

if ($Target -eq "gui" -or $Target -eq "all") {
    Stop-Process -Name "gui_qt" -Force -ErrorAction SilentlyContinue
    Write-Host "[build] Building gui_qt.exe (Qt Widgets)..." -ForegroundColor Cyan
    & g++ $FLAGS $INCL $QTINC $LIB demos/gui_qt.cpp $QTLIBS -o gui_qt.exe
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] gui_qt.exe build failed. Ensure window is closed." -ForegroundColor Yellow
    } else {
        Write-Host "[build] OK: gui_qt.exe" -ForegroundColor Green
    }
}

if ($Target -eq "all" -or $Target -eq "demos") {
    Build-Target "playground" "demos/playground.cpp"
    Build-Target "xor_demo" "demos/xor_demo.cpp"
    Build-Target "and_or_demo" "demos/and_or_demo.cpp"
    Build-Target "iris_demo" "demos/iris_demo.cpp"
    Build-Target "compare_demo" "demos/compare_demo.cpp"
    Build-Target "early_stop_demo" "demos/early_stop_demo.cpp"
}

if ($Target -eq "all" -or $Target -eq "tests") {
    Build-Target "gradient_check" "tests/gradient_check.cpp"
    Build-Target "test_basic" "tests/test_basic.cpp"
    Build-Target "test_choices" "tests/test_choices.cpp"
    Build-Target "test_training" "tests/test_training.cpp"
    Build-Target "test_correctness" "tests/test_correctness.cpp"
    Build-Target "test_callbacks" "tests/test_callbacks.cpp"
    Build-Target "test_patterns" "tests/test_patterns.cpp"
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host " Build completed successfully!      " -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
