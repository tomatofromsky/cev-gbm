<#
.SYNOPSIS
    Reproduce a single experiment end-to-end (PowerShell mirror of run_experiment.sh).

.DESCRIPTION
    Pick an experiment with -Target:
      gbm — GBM forward process (alpha=1, log-prices). Paper's headline.
      vp  — VP SDE baseline on log-returns (alpha=0).
      ve  — VE SDE baseline on log-returns (alpha=0).

    Each target sets sensible defaults for Alpha / Sde / Schedule. Override any
    parameter explicitly on the command line OR via the matching environment
    variable (TARGET, ALPHA, SDE, SCHEDULE, BATCH_SIZE, EPOCHS, LR, SIGMA_MIN,
    SIGMA_MAX, NUM_SAMPLES, STEPS, SNR, N_CORR, CUDA_VISIBLE_DEVICES).

    Idempotency:
      - data prep is skipped if data\sp500_subseq*.pt already exist
      - training auto-resumes from the highest-numbered checkpoint
      - sampling and plotting overwrite their output folders

.EXAMPLE
    ./run_experiment.ps1
    # Default: TARGET=gbm (paper headline)

.EXAMPLE
    ./run_experiment.ps1 -Target ve
    # VE baseline, exponential schedule

.EXAMPLE
    ./run_experiment.ps1 -Target gbm -Schedule exponential
    # GBM with a different schedule

.EXAMPLE
    ./run_experiment.ps1 -Epochs 2 -BatchSize 16 -Steps 100 -NumSamples 5
    # Quick smoke

.EXAMPLE
    $env:TARGET = 've'; ./run_experiment.ps1
    # Env-var style (parity with the bash script)
#>

[CmdletBinding()]
param(
    [string]$Target,
    [Nullable[int]]$Alpha,
    [string]$Sde,
    [string]$Schedule,
    [Nullable[double]]$SigmaMin,
    [Nullable[double]]$SigmaMax,
    [Nullable[int]]$BatchSize,
    [Nullable[int]]$Epochs,
    [string]$Lr,
    [Nullable[int]]$NumSamples,
    [Nullable[int]]$Steps,
    [Nullable[double]]$Snr,
    [Nullable[int]]$NCorr,
    [string]$CudaVisibleDevices,
    [Nullable[int]]$NumGpus
)

$ErrorActionPreference = 'Stop'

function Banner {
    param([string]$Text)
    Write-Host ''
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

# Resolve a value: explicit param wins; else env var; else fallback default.
function Resolve-Default {
    param($Value, [string]$EnvName, $Fallback)
    if ($null -ne $Value -and "$Value" -ne '') { return $Value }
    $envItem = Get-Item "env:$EnvName" -ErrorAction SilentlyContinue
    if ($envItem -and $envItem.Value) { return $envItem.Value }
    return $Fallback
}

# --- Target selection ------------------------------------------------------
$Target = Resolve-Default $Target 'TARGET' 'gbm'
$Target = $Target.ToLower()

$targetDefaults = @{
    # gbm: per paper §3 (eq. 3.5), GBM forward process reduces to VE SDE in
    # log-price space when μ_t = ½σ_t², so the matching loss branch is VE.
    # Schedule defaults to exponential — paper §4 reports α≈4.62 (closest to
    # empirical α≈4.35); cosine and linear are also valid.
    'gbm' = @{ Alpha = 1; Sde = 'VE'; Schedule = 'exponential' }
    'vp'  = @{ Alpha = 0; Sde = 'VP'; Schedule = 'cosine' }
    've'  = @{ Alpha = 0; Sde = 'VE'; Schedule = 'exponential' }
}
if (-not $targetDefaults.ContainsKey($Target)) {
    Write-Error "Unknown -Target $Target; valid: gbm, vp, ve"
    exit 1
}

$Alpha    = [int]   (Resolve-Default $Alpha    'ALPHA'    $targetDefaults[$Target].Alpha)
$Sde      =         (Resolve-Default $Sde      'SDE'      $targetDefaults[$Target].Sde)
$Schedule =         (Resolve-Default $Schedule 'SCHEDULE' $targetDefaults[$Target].Schedule)

# --- Other knobs (paper §4 defaults) ---------------------------------------
$SigmaMin   = [double](Resolve-Default $SigmaMin   'SIGMA_MIN'  0.01)
$SigmaMax   = [double](Resolve-Default $SigmaMax   'SIGMA_MAX'  1.0)
$BatchSize  = [int]   (Resolve-Default $BatchSize  'BATCH_SIZE' 64)
$Epochs     = [int]   (Resolve-Default $Epochs     'EPOCHS'     1000)
$Lr         =         (Resolve-Default $Lr         'LR'         '1e-4')
$NumSamples = [int]   (Resolve-Default $NumSamples 'NUM_SAMPLES' 120)
$Steps      = [int]   (Resolve-Default $Steps      'STEPS'      2000)
$Snr        = [double](Resolve-Default $Snr        'SNR'        0.2)
$NCorr      = [int]   (Resolve-Default $NCorr      'N_CORR'     1)
$NumGpus    = [int]   (Resolve-Default $NumGpus    'NUM_GPUS'   1)
$CudaVisibleDevices = (Resolve-Default $CudaVisibleDevices 'CUDA_VISIBLE_DEVICES' '0')

# --- Auto-derived paths (mirrors train.py's preset logic) ------------------
# Schedule is baked into ModelDir so different schedules don't collide.
if ($Alpha -eq 1) {
    $ModelDir = "save_model_bs_${Schedule}_64"
    $Tag      = "BS_${Schedule}"
} elseif ($Alpha -eq 0 -and $Sde -eq 'VE') {
    $ModelDir = "save_model_ve_${Schedule}_64"
    $Tag      = "VE_${Schedule}"
} elseif ($Alpha -eq 0 -and $Sde -eq 'VP') {
    $ModelDir = "save_model_vp_${Schedule}_64"
    $Tag      = "VP_${Schedule}"
} else {
    Write-Error "Unsupported (Alpha, Sde) = ($Alpha, $Sde)"
    exit 1
}

$PlotDir = "${Tag}_plot_"
$CsvDir  = "${Tag}_csv_"
$TsDir   = "${Tag}_timeseries"

# --- Activate venv ---------------------------------------------------------
Banner 'Activating venv'
$activateCandidates = @(
    '.venv/Scripts/Activate.ps1',   # Windows venv layout
    '.venv/bin/Activate.ps1'        # Linux/macOS venv layout
)
$activateScript = $activateCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $activateScript) {
    Write-Error "Could not find venv activation script under .venv/. Create it with: python -m venv .venv"
    exit 1
}
. $activateScript

# Pin GPU for child processes
$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices

# --- 1. Data prep ----------------------------------------------------------
Banner 'Data prep'
$dataReady = (Test-Path 'data/sp500_subseq.pt') -and `
             (Test-Path 'data/sp500_subseq_log.pt') -and `
             (Test-Path 'data/sp500_subseq_original.pt')

if ($dataReady) {
    Write-Host 'data/sp500_subseq*.pt already present — skipping download.'
} else {
    New-Item -ItemType Directory -Force -Path 'data' | Out-Null
    python data_download.py
    if ($LASTEXITCODE -ne 0) { Write-Error "data_download.py failed (exit $LASTEXITCODE)"; exit 1 }
    Get-ChildItem -Path 'sp500_subseq*.pt' -ErrorAction SilentlyContinue | Move-Item -Destination 'data/' -Force
    Get-ChildItem -Path 'global_*.pkl'      -ErrorAction SilentlyContinue | Move-Item -Destination 'data/' -Force
}

# --- 2. Training -----------------------------------------------------------
# Multi-GPU via torchrun when NumGpus > 1; plain python otherwise.
if ($NumGpus -gt 1) {
    $effectiveBatch = $BatchSize * $NumGpus
    Banner "Training: target=$Target (alpha=$Alpha, sde=$Sde, schedule=$Schedule) epochs=$Epochs | DDP nproc=$NumGpus effective_batch=$effectiveBatch"
    torchrun --standalone --nnodes=1 --nproc_per_node=$NumGpus train.py `
        --alpha $Alpha --sde $Sde --noise_schedule $Schedule `
        --batch_size $BatchSize --epochs $Epochs --lr $Lr `
        --sigma_min $SigmaMin --sigma_max $SigmaMax
} else {
    Banner "Training: target=$Target (alpha=$Alpha, sde=$Sde, schedule=$Schedule) epochs=$Epochs | single-GPU batch=$BatchSize"
    python train.py `
        --alpha $Alpha --sde $Sde --noise_schedule $Schedule `
        --batch_size $BatchSize --epochs $Epochs --lr $Lr `
        --sigma_min $SigmaMin --sigma_max $SigmaMax
}
if ($LASTEXITCODE -ne 0) { Write-Error "train.py failed (exit $LASTEXITCODE)"; exit 1 }

# Pick the highest-numbered checkpoint that actually exists.
$Ckpt = Get-ChildItem -Path $ModelDir -Filter 'model_epoch_*.pth' -ErrorAction SilentlyContinue |
    Sort-Object { [int]([regex]::Match($_.BaseName, 'model_epoch_(\d+)').Groups[1].Value) } -Descending |
    Select-Object -First 1
if (-not $Ckpt) {
    Write-Error "No checkpoint produced under $ModelDir/"
    exit 1
}
Write-Host "Using checkpoint: $($Ckpt.FullName)"

# --- 3. Sampling + stylized-fact plots -------------------------------------
Banner "Sampling $NumSamples sequences ($Steps reverse-diffusion steps)"
python plot_result.py `
    --model_path $Ckpt.FullName `
    --alpha $Alpha --sde $Sde --noise_schedule $Schedule `
    --sigma_min $SigmaMin --sigma_max $SigmaMax `
    --num_samples $NumSamples --steps $Steps --snr $Snr --n_corr $NCorr `
    --plot_folder $PlotDir `
    --csv_folder $CsvDir `
    --plot_timeseries $TsDir
if ($LASTEXITCODE -ne 0) { Write-Error "plot_result.py failed (exit $LASTEXITCODE)"; exit 1 }

# --- 4. Real-data baseline -------------------------------------------------
Banner 'Real-data baseline'
$oriPopulated = (Test-Path 'ori_plot') -and `
                ((Get-ChildItem 'ori_plot' -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
if ($oriPopulated) {
    Write-Host 'ori_plot/ already populated — skipping.'
} else {
    python ori_plot_result.py `
        --data_file 'data/sp500_subseq_original.pt' `
        --num_samples 1000
    if ($LASTEXITCODE -ne 0) { Write-Error "ori_plot_result.py failed (exit $LASTEXITCODE)"; exit 1 }
}

# --- 5. Done ---------------------------------------------------------------
Banner 'Done'
@"
Synthetic stylized-fact plots:  $PlotDir/
Per-sample log-return CSVs:     $CsvDir/
Recovered price series PNGs:    $TsDir/
Real-data baseline plots:       ori_plot/

Compare the matching pair of files in $PlotDir/ and ori_plot/ side by side:
  generated_metrics_heavy_tail.png
  generated_metrics_volatility_clustering.png
  generated_metrics_leverage_effect.png
  generated_metrics_linear_unpredictability.png
  generated_metrics_coarse_fine.png
  generated_metrics_gain_loss_asymmetry.png
"@ | Write-Host
