$ErrorActionPreference = "Stop"

$Repo = "isartor-ai/Isartor"
$BinName = "isartor.exe"
$Target = "x86_64-pc-windows-msvc"
$InstallDir = "$env:USERPROFILE\.local\bin"

Write-Host "Installing $BinName..."

# Fetch latest release tag
$LatestReleaseUrl = "https://api.github.com/repos/$Repo/releases/latest"
Write-Host "Fetching latest release information..."
$ReleaseData = Invoke-RestMethod -Uri $LatestReleaseUrl
$Tag = $ReleaseData.tag_name

if (-Not $Tag) {
    Write-Error "Could not determine the latest release tag."
    exit 1
}

$Archive = "isartor-${Tag}-${Target}.zip"
$DownloadUrl = "https://github.com/$Repo/releases/download/$Tag/$Archive"

Write-Host "Downloading $Archive from $DownloadUrl ..."

$TmpDir = Join-Path $env:TEMP "isartor-install-$([System.IO.Path]::GetRandomFileName())"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

$ArchivePath = Join-Path $TmpDir $Archive
Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath

Write-Host "Extracting..."
Expand-Archive -Path $ArchivePath -DestinationPath $TmpDir -Force

if (-Not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}

$ExtractedBin = Get-ChildItem -Path $TmpDir -Recurse -Filter $BinName -File | Select-Object -First 1
if (-not $ExtractedBin) {
    Write-Error "Could not find $BinName in the extracted archive."
    Remove-Item -Recurse -Force $TmpDir
    exit 1
}

$DestPath = Join-Path $InstallDir $BinName
Copy-Item -Path $ExtractedBin.FullName -Destination $DestPath -Force

Remove-Item -Recurse -Force $TmpDir

Write-Host "✅ $BinName $Tag installed to $InstallDir!"

# Add to PATH if not already present
$Path = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($Path -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$Path;$InstallDir", "User")
    $env:PATH = "$env:PATH;$InstallDir"
    Write-Host "Added $InstallDir to your PATH."
}

Write-Host ""
Write-Host "Quick start:"
Write-Host "  isartor          -- start the server (port 8080)"
Write-Host "  isartor demo     -- run the deflection demo (no API key needed)"
Write-Host "  isartor init     -- generate a config scaffold"
