param(
    [Parameter(Mandatory=$true)][string]$SourceDir,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [int]$Jobs=4
)
$ErrorActionPreference='Stop'
$sourcePath=(Resolve-Path -LiteralPath $SourceDir).Path
$commit=(& git -C $sourcePath rev-parse HEAD)
if($LASTEXITCODE -ne 0 -or $commit -ne 'd327bf1f0780f87b8c01b367b61853b0c291d7c4'){throw 'Expected b11869 source'}
foreach($name in @('cmake','ninja','cl','icx')){if(-not (Get-Command $name -ErrorAction SilentlyContinue)){throw "Missing $name; initialize VS x64 and oneAPI environments first"}}
if(-not $env:DNNLROOT){throw 'Set DNNLROOT to your installed oneDNN directory'}
if(-not $env:LEVEL_ZERO_V1_SDK_PATH){throw 'Set LEVEL_ZERO_V1_SDK_PATH to your installed Level Zero SDK'}
& cmake -S $sourcePath -B $BuildDir -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_CPU=OFF -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF
if($LASTEXITCODE -ne 0){throw 'CMake configure failed'}
& cmake --build $BuildDir --target ggml-sycl test-backend-ops -j $Jobs
if($LASTEXITCODE -ne 0){throw 'SYCL build failed'}
Write-Output 'Build complete. Use only with the matching b11869 runtime; see setup-windows.md for staging and numerical tests.'
