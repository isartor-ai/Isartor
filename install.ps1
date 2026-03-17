$ErrorActionPreference = "Stop"

$Repo = if ($env:ISARTOR_REPO) { $env:ISARTOR_REPO } else { "isartor-ai/Isartor" }
$BinName = "isartor.exe"
$Target = "x86_64-pc-windows-msvc"
$InstallDir = if ($env:ISARTOR_INSTALL_DIR) { $env:ISARTOR_INSTALL_DIR } else { "$env:USERPROFILE\.local\bin" }

$Token = if ($env:ISARTOR_GITHUB_TOKEN) { $env:ISARTOR_GITHUB_TOKEN } elseif ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $env:GH_TOKEN }

function Get-GitHubHeaders([string]$Accept) {
    $h = @{
        "Accept" = $Accept
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "isartor-installer"
    }

    if ($Token) {
        $h["Authorization"] = "Bearer $Token"
    }

    return $h
}

Write-Host "Installing $BinName from $Repo..."

# Fetch latest release tag
$LatestReleaseUrl = "https://api.github.com/repos/$Repo/releases/latest"
Write-Host "Fetching latest release information..."

try {
    $ReleaseData = Invoke-RestMethod -Uri $LatestReleaseUrl -Headers (Get-GitHubHeaders "application/vnd.github+json")
} catch {
    Write-Error "Could not fetch the latest release from GitHub."
    Write-Host ""
    Write-Host "If $Repo is a private repository, authenticate first:"
    Write-Host "  gh auth login"
    Write-Host "  gh api -H \"Accept: application/vnd.github.raw\" /repos/$Repo/contents/install.ps1?ref=main | iex"
    Write-Host ""
    Write-Host "Or set a token (needs repo scope for private repos):"
    Write-Host "  setx GITHUB_TOKEN <token>"
    throw
}

$Tag = $ReleaseData.tag_name
if (-Not $Tag) {
    Write-Error "Could not determine the latest release tag."
    exit 1
}

$Archive = "isartor-${Tag}-${Target}.zip"
$TmpDir = Join-Path $env:TEMP "isartor-install-$([System.IO.Path]::GetRandomFileName())"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

$ArchivePath = Join-Path $TmpDir $Archive

$Asset = $null
if ($ReleaseData.assets) {
    $Asset = $ReleaseData.assets | Where-Object { $_.name -eq $Archive } | Select-Object -First 1
}

if ($Asset -and $Token) {
    Write-Host "Downloading $Archive via GitHub API ..."
    Invoke-WebRequest -Uri $Asset.url -Headers (Get-GitHubHeaders "application/octet-stream") -OutFile $ArchivePath
} else {
    $DownloadUrl = "https://github.com/$Repo/releases/download/$Tag/$Archive"
    Write-Host "Downloading $Archive from $DownloadUrl ..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath
}

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
