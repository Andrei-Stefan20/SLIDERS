<#
.SYNOPSIS
  Run the full SLIDERS pipeline sequentially, stopping at the first failure.

.DESCRIPTION
  Mirrors docs/GUIDE.md: train SAE -> name features -> build index -> class
  directions -> evaluate -> report. Embedding extraction and the UI are opt-in
  (the embeddings rarely change; the UI blocks). Uses .venv\Scripts\python.exe if
  present, otherwise whatever `python` is on PATH.

  -Patches switches to the patch-level pipeline: DINOv2 patch tokens + a patch SAE +
  late-interaction MaxSim retrieval/evaluation. Artifacts use the "<Dataset>_patch"
  stem; class directions are skipped (they are CLS-only).

.EXAMPLE
  .\scripts\run_pipeline.ps1
  Retrain and run everything through the report, reusing existing CLS embeddings.

.EXAMPLE
  .\scripts\run_pipeline.ps1 -Patches -ExtractEmbeddings
  Extract patch tokens (int8), train the patch SAE, name, index, and evaluate (MaxSim).

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
  [switch]$Patches,
  [ValidateSet("float16", "float32", "int8")][string]$PatchDtype = "int8",
  [int]$MaxPatchesPerImage = 0,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$d = "data/processed"
$m = "models"
$valDataset = $Dataset -replace "_train$", "_val"

# Patch artifacts carry a "_patch" stem; CLS artifacts use the dataset name as-is.
$stem = if ($Patches) { "${Dataset}_patch" } else { $Dataset }
$valStem = if ($Patches) { "${valDataset}_patch" } else { $valDataset }
$evalJson = "reports/${stem}_eval.json"

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
  $patchFlags = @()
  if ($Patches) {
    $patchFlags = @("--use-patches", "--patch-dtype", $PatchDtype)
    if ($MaxPatchesPerImage -gt 0) { $patchFlags += @("--max-patches-per-image", "$MaxPatchesPerImage") }
  }
  Step "[0a] Extract train embeddings" (@(
    "scripts/extract_embeddings.py", "--config", $Config, "--dataset", $Dataset) + $patchFlags)
  Step "[0b] Extract val embeddings" (@(
    "scripts/extract_embeddings.py", "--config", $Config,
    "--dataset", $valDataset, "--input", $ValInput) + $patchFlags)
}

Step "[1] Train SAE" @(
  "scripts/train_sae.py", "--embeddings", "$d/${stem}_embeddings.npy",
  "--output", "$m/", "--config", $Config, "--topk", "$Topk")

Step "[2] Name features" @(
  "scripts/name_features.py", "--config", $Config,
  "--embeddings", "$d/${stem}_embeddings.npy",
  "--image-paths", "$d/${stem}_image_paths.json",
  "--sae-model", "$m/${stem}_sae_best.pt")

$idxArgs = @("scripts/build_index.py", "--embeddings", "$d/${stem}_embeddings.npy")
if (-not $Patches) { $idxArgs += @("--sae-model", "$m/${stem}_sae_best.pt") }
Step "[3] Build index" $idxArgs

if (-not $SkipClassDirections -and -not $Patches) {
  Step "[4] Compute class directions" @(
    "scripts/compute_class_directions.py",
    "--embeddings", "$d/${stem}_embeddings.npy",
    "--image-paths", "$d/${stem}_image_paths.json",
    "--adapter", "plantvillage", "--output", "$d/")
}

$evalArgs = @(
  "scripts/evaluate.py",
  "--embeddings", "$d/${stem}_embeddings.npy",
  "--image-paths", "$d/${stem}_image_paths.json",
  "--index", "$d/${stem}_index.faiss",
  "--sae-model", "$m/${stem}_sae_best.pt",
  "--query-embeddings", "$d/${valStem}_embeddings.npy",
  "--query-image-paths", "$d/${valStem}_image_paths.json",
  "--output", $evalJson)
if (-not $Patches) {
  $evalArgs += @(
    "--feature-names", "$m/${stem}_feature_names.json",
    "--class-directions", "$d/${stem}_class_directions.npy")
}
Step "[5] Evaluate (held-out)" $evalArgs

Step "[6] Generate report" @(
  "scripts/make_report.py", "--dataset", $stem, "--eval-json", $evalJson)

$elapsed = (Get-Date) - $start
Write-Host "`nPipeline finished in $([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s." -ForegroundColor Green
Write-Host "Report: reports/$stem/" -ForegroundColor Green

if ($StartUI) {
  $uiArgs = if ($Patches) {
    @("-m", "src.api", "--dataset", $stem, "--host", "127.0.0.1", "--port", "8000")
  } else {
    @("-m", "src.api", "--config", $Config, "--host", "127.0.0.1", "--port", "8000")
  }
  Step "[7] Start UI" $uiArgs
}
