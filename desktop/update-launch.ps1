param(
  [Parameter(Mandatory = $true)][string]$CommandLine
)

$ErrorActionPreference = "Stop"

try {
  $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
  }
  if ($result.ReturnValue -ne 0) {
    throw "Win32_Process.Create devolvio $($result.ReturnValue)"
  }
  Write-Output ("updater pid=" + $result.ProcessId)
  exit 0
} catch {
  Write-Error ("No se pudo iniciar el actualizador: " + $_.Exception.Message)
  exit 1
}
