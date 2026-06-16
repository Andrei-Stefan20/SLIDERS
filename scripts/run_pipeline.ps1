<#
.SYNOPSIS
  Run the full SLIDERS pipeline sequentially, stopping at the first failure.

.DESCRIPTION
  Mirrors docs/GUIDE.md: train SAE -> name features -> build index -> class
  directions -> evaluate -> report. Embedding extraction and the UI are opt-in
  (the embeddings rarely change; the UI blocks). Uses .venv\Scripts\python.exe if
  present, otherwise whatever `python` is on PATH.

.EXAMPLE
  .\scripts\run_pipeline.ps1
  Retrain and run everything through the report, reusing existing embeddings.

.EXAMPLE
  .\scripts\run_pipeline.ps1 -ExtractEmbeddings -StartUI
  Also re-extract embeddings first and launch the UI at the end.

.EXAMPLE
  .\scripts\run_pipeline.ps1 -DryRun
  Print the commands without running them.
#>
param(
  [string]$Dataset = "plantvillage_train",
  [string]$Config = "configs/plantvillage.yaml",
  [int]$Topk = 40,
  [string]$ValInput = "data/raw/plantvillage/PlantVillage/val",
  [switch]$ExtractEmbeddings,
  [switch]$SkipClassDirections,
  [switch]$StartUI,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$d = "data/processed"
$m = "models"
$valDataset = $Dataset -replace "_train$", "_val"
$evalJson = "reports/${Dataset}_eval.json"

$Py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

function Step {
  param([string]$Name, [string[]]$CmdArgs)
  Write-Host "`n=== $Name ===" -ForegroundColor Cyan
  Write-Host "$Py $($CmdArgs -join ' ')" -ForegroundColor DarkGray
  if ($DryRun) { return }
  & $Py @CmdArgs
  if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit code $LASTEXITCODE)" }
}

$start = Get-Date

if ($ExtractEmbeddings) {
  Step "[0a] Extract train embeddings" @(
    "scripts/extract_embeddings.py", "--config", $Config, "--dataset", $Dataset)
  Step "[0b] Extract val embeddings" @(
    "scripts/extract_embeddings.py", "--config", $Config,
    "--dataset", $valDataset, "--input", $ValInput)
}

Step "[1] Train SAE" @(
  "scripts/train_sae.py", "--embeddings", "$d/${Dataset}_embeddings.npy",
  "--output", "$m/", "--config", $Config, "--topk", "$Topk")

Step "[2] Name features" @(
  "scripts/name_features.py", "--config", $Config,
  "--embeddings", "$d/${Dataset}_embeddings.npy",
  "--image-paths", "$d/${Dataset}_image_paths.json",
  "--sae-model", "$m/${Dataset}_sae_best.pt")

Step "[3] Build index + activations" @(
  "scripts/build_index.py", "--embeddings", "$d/${Dataset}_embeddings.npy",
  "--sae-model", "$m/${Dataset}_sae_best.pt")

if (-not $SkipClassDirections) {
  Step "[4] Compute class directions" @(
    "scripts/compute_class_directions.py",
    "--embeddings", "$d/${Dataset}_embeddings.npy",
    "--image-paths", "$d/${Dataset}_image_paths.json",
    "--adapter", "plantvillage", "--output", "$d/")
}

Step "[5] Evaluate (held-out)" @(
  "scripts/evaluate.py",
  "--embeddings", "$d/${Dataset}_embeddings.npy",
  "--image-paths", "$d/${Dataset}_image_paths.json",
  "--index", "$d/${Dataset}_index.faiss",
  "--sae-model", "$m/${Dataset}_sae_best.pt",
  "--query-embeddings", "$d/${valDataset}_embeddings.npy",
  "--query-image-paths", "$d/${valDataset}_image_paths.json",
  "--feature-names", "$m/${Dataset}_feature_names.json",
  "--class-directions", "$d/${Dataset}_class_directions.npy",
  "--output", $evalJson)

Step "[6] Generate report" @(
  "scripts/make_report.py", "--dataset", $Dataset, "--eval-json", $evalJson)

$elapsed = (Get-Date) - $start
Write-Host "`nPipeline finished in $([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s." -ForegroundColor Green
Write-Host "Report: reports/$Dataset/" -ForegroundColor Green

if ($StartUI) {
  Step "[7] Start UI" @("-m", "src.api", "--config", $Config, "--host", "127.0.0.1", "--port", "8000")
}
