param(
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [int]$Seconds=240,
    [int]$Interval=2
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Output already exists' }
$until = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $until) {
    $processes = @(Get-Process -Name llama-server -ErrorAction SilentlyContinue)
    $ids = @($processes | ForEach-Object { $_.Id })
    $counters = Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage','\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue
    $samples = @($counters.CounterSamples | Where-Object {
        $_.InstanceName -match '^pid_(\d+)_' -and [int]$Matches[1] -in $ids
    } | ForEach-Object { @{instance=$_.InstanceName; path=$_.Path; value=$_.CookedValue} })
    $row = @{utc=(Get-Date).ToUniversalTime().ToString('o'); processes=@($processes | ForEach-Object {
        @{pid=$_.Id; working_set=$_.WorkingSet64; private_bytes=$_.PrivateMemorySize64}
    }); counters=$samples}
    $json = ConvertTo-Json -InputObject $row -Depth 5 -Compress
    [IO.File]::AppendAllText($OutputPath, $json+[char]10, [Text.UTF8Encoding]::new($false))
    Start-Sleep -Seconds $Interval
}
