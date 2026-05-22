$base = 'C:\LAB\local-intelligent-system'
$dirs = 'core\agents','core\tools','core\memory','core\rag','data\documents','data\vector_db','data\long_term','models','frontend'
foreach ($d in $dirs) { New-Item -ItemType Directory -Path "$base\$d" -Force | Out-Null }
Write-Host "All directories created."
