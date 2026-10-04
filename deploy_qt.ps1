# deploy_qt.ps1 — make gui_qt.exe double-clickable on any machine.
# Problem: Windows resolves Qt6 DLLs via PATH, and MiKTeX ships its own
# older Qt6 (6.8.3) which shadows MSYS2 UCRT64 Qt (6.11.x) -> instant death.
# Fix: copy the exact ucrt64 DLLs + platform plugin next to the exe.
# Windows searches the exe's own folder FIRST, so local copies always win.
#
# Usage: powershell -ExecutionPolicy Bypass -File .\deploy_qt.ps1
param(
    [string]$QtBin = 'C:\msys64\ucrt64\bin',
    [string]$QtPlugins = 'C:\msys64\ucrt64\share\qt6\plugins'
)
$ErrorActionPreference = 'Stop'
# ldd follows the same DLL search order as the Windows loader, so force it
# to see OUR Qt first (otherwise it resolves MiKTeX's Qt and finds nothing).
$env:Path = "$QtBin;" + $env:Path
# ldd follows the same DLL search order as the Windows loader, so force it
# to see OUR Qt first (otherwise it resolves MiKTeX's Qt and finds nothing).
$env:Path = "$QtBin;" + $env:Path
$Ldd = 'C:\msys64\usr\bin\ldd.exe'
if (!(Test-Path $Ldd)) { throw "ldd not found at $Ldd (install msys2-base?)" }
if (!(Test-Path '.\gui_qt.exe')) { throw 'Run this from MiniANN_MVP (gui_qt.exe missing — build first).' }

# Recursively collect every /ucrt64/bin/*.dll dependency via ldd.
$seen = @{}
$queue = @((Resolve-Path '.\gui_qt.exe').Path)
while ($queue.Count -gt 0) {
    $bin = $queue[0]
    if ($queue.Count -gt 1) { $queue = $queue[1..($queue.Count - 1)] } else { $queue = @() }
    & $Ldd $bin 2>$null | ForEach-Object {
        if ($_ -match '=>\s+(/ucrt64/bin/[^ ]+\.dll)') {
            $name = Split-Path $Matches[1] -Leaf
            if (!$seen.ContainsKey($name)) {
                $seen[$name] = $Matches[1]
                $queue += (Join-Path $QtBin $name)
            }
        }
    }
}
Write-Host ("Found {0} ucrt64 DLL dependencies." -f $seen.Count)
foreach ($n in ($seen.Keys | Sort-Object)) {
    $src = Join-Path $QtBin $n
    if ((Test-Path $src) -and !(Test-Path ".\$n")) {
        Copy-Item -LiteralPath $src -Destination ".\$n"
        Write-Host "  copied $n"
    }
}

# Platform plugin: Qt searches <exeDir>/platforms automatically (no qt.conf needed).
New-Item -ItemType Directory -Path '.\platforms' -Force | Out-Null
foreach ($plug in @('qwindows.dll', 'qminimal.dll', 'qoffscreen.dll')) {
    $src = Join-Path $QtPlugins "platforms\$plug"
    if ((Test-Path $src) -and !(Test-Path ".\platforms\$plug")) {
        Copy-Item -LiteralPath $src -Destination ".\platforms\$plug"
        Write-Host "  copied platforms\$plug"
    }
}
Write-Host 'Deploy done. gui_qt.exe now uses local Qt regardless of PATH.'
