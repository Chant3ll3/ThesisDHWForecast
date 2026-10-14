"""
config.py — Single source of truth for all project parameters.

All four phases (data preparation, model training, model inference,
results analysis) read their settings from the dataclasses defined here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataPreparationConfig:
    """Parameters for Phase 1: Data Preparation."""

    # ── Input ────────────────────────────────────────────────────
    input_dir: str = "/g/data/ia39/aus-ref-clim-data-nci/oisst/data/yearly/"
    file_pattern: str = "oisst-avhrr-v02r01_{year}.nc"
    year_range: Tuple[int, int] = (1981, 2024)
    variable: str = "sst"

    # ── Detrending ───────────────────────────────────────────────
    # "global_linear" — fit a single linear trend to the global
    # (area-weighted) mean SST time series, then subtract that
    # trend from every grid point so spatial relationships are kept.
    detrend_method: str = "global_linear"

    # ── Standardisation ──────────────────────────────────────────
    # "global" — compute a single mean and std over all valid
    # (time, lat, lon) values and standardise uniformly.
    standardise_method: str = "global"

    # ── Land / missing-value fill ────────────────────────────────
    # SFNO requires a complete field (no NaN). Land points must be
    # filled smoothly to avoid spectral ringing.
    # "diffusion"  — iterative Laplacian relaxation (extrapolates
    #                ocean values smoothly onto land).
    # "nearest"    — scipy nearest-neighbour fill (fast but blocky).
    # "zero"       — simple zero fill (creates discontinuities).
    fill_method: str = "diffusion"
    fill_diffusion_iterations: int = 500    # more iterations = smoother
    fill_fallback_value: float = 0.0        # fills any remaining NaN after diffusion

    # ── Spatial downscaling ──────────────────────────────────────
    downscale: bool = True
    target_lon_size: int = 128
    target_lat_size: int = 64

    # ── Region subsetting (applied *after* standardisation) ──────
    subset_region: bool = False
    lat_bounds: Optional[Tuple[float, float]] = None   # e.g. (-50.0, 0.0)
    lon_bounds: Optional[Tuple[float, float]] = None   # e.g. (100.0, 180.0)

    # ── Diagnostics ──────────────────────────────────────────────
    diagnostic_year: int = 2020          # year used for the 12-month plots
    diagnostic_pdf: str = "diagnostics/data_preparation_diagnostics.pdf"

    # ── Output ───────────────────────────────────────────────────
    output_dir: str = "data/processed/"
    save_intermediates: bool = True      # cheap after downscale (128×64)
    downscaled_file: str = "sst_downscaled.nc"
    detrended_file: str = "sst_detrended.nc"
    standardised_file: str = "sst_standardised.nc"
    filled_file: str = "sst_filled.nc"
    final_file: str = "sst_final.nc"
    subset_file: str = "sst_subset.nc"
    trend_params_file: str = "trend_params.nc"       # slope & intercept
    standardise_params_file: str = "std_params.nc"   # mean & std

    # ── Resume ─────────────────────────────────────────────────
    # Set to the last COMPLETED stage number to skip earlier stages.
    #   0 = run from scratch (default)
    #   1 = resume after load            (reload raw)
    #   2 = resume after downscale       (reload downscaled)
    #   3 = resume after detrend         (reload detrended)
    #   4 = resume after standardise     (reload standardised)
    #   5 = resume after fill            (reload filled)
    resume_after_stage: int = 0

    # ── Performance ──────────────────────────────────────────────
    chunks: dict = field(default_factory=lambda: {"time": 366, "lat": 720, "lon": 720})
    # Dask scheduler: "distributed" (LocalCluster), "threads", or "synchronous"
    # "threads" is best for the downscaled grid (small data, no IPC overhead).
    # "distributed" is useful for full-resolution processing or if you want
    # the dask dashboard.
    dask_scheduler: str = "threads"
    dask_n_workers: int = 8             # for distributed scheduler only
    dask_threads_per_worker: int = 2    # for distributed scheduler only
    dask_memory_limit: str = "12GB"     # for distributed scheduler only
    dask_num_threads: int = 16          # for threaded scheduler


@dataclass
class ModelTrainingConfig:
    """Parameters for Phase 2: Model Training (SFNO for SST forecasting)."""

    # ── SFNO Architecture ────────────────────────────────────────
    # Input/output: number of SST fields fed as channels
    input_history: int = 3              # past days as input channels (t, t-1, t-2)
    output_channels: int = 1            # predict single SST field at t+1
    embed_dim: int = 64                 # embedding / hidden channel width
    num_layers: int = 4                 # number of SFNO blocks
    num_modes_lat: int = 16             # spectral truncation in latitude
    num_modes_lon: int = 32             # spectral truncation in longitude
    mlp_ratio: float = 2.0              # MLP hidden dim = embed_dim * mlp_ratio
    activation: str = "gelu"            # "gelu", "relu", "silu"
    norm_type: str = "layer_norm"       # "layer_norm", "instance_norm", "none"
    dropout_rate: float = 0.2
    use_residual: bool = True           # global skip from input to output
    spectral_bias: bool = True          # learnable bias in spectral domain

    # ── Positional / geographic encoding ─────────────────────────
    # Append lat/lon grids as extra input channels so the model
    # is aware of geographic location.
    add_grid_features: bool = True      # adds 2 extra input channels (lat, lon)

    # ── Forecasting strategy ─────────────────────────────────────
    forecast_horizon: int = 90          # maximum lead time in days
    # Autoregressive rollout during training (gradually increased)
    initial_rollout_steps: int = 1      # start training with 1-step prediction
    max_rollout_steps: int = 15         # ramp up to this many rollout steps
    rollout_epoch_start: int = 10       # epoch at which rollout ramping begins
    rollout_epoch_end: int = 60         # epoch by which max rollout is reached
    rollout_loss_decay: float = 0.98    # exponential weight decay per rollout step

    # ── Data splitting ───────────────────────────────────────────
    train_years: Tuple[int, int] = (1982, 2015)
    val_years: Tuple[int, int] = (2016, 2019)
    test_years: Tuple[int, int] = (2020, 2024)

    # ── Training hyperparameters ─────────────────────────────────
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    batch_size: int = 8
    num_epochs: int = 100
    weight_decay: float = 1e-5
    optimizer: str = "adamw"            # "adam", "adamw", "sgd"
    scheduler: str = "cosine"           # "cosine", "plateau", "step", "none"
    scheduler_patience: int = 10        # for plateau scheduler
    scheduler_factor: float = 0.5       # for plateau scheduler
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0     # max gradient norm (0 = disabled)
    loss_function: str = "mse"          # "mse", "mae", "huber"

    # ── Early stopping ───────────────────────────────────────────
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_loss"

    # ── Paths ────────────────────────────────────────────────────
    input_data_path: str = "data/processed/sst_final.nc"
    checkpoint_dir: str = "checkpoints/"
    best_model_path: str = "checkpoints/best_model.pt"
    training_log_path: str = "results/training_log.csv"

    # ── Reproducibility ──────────────────────────────────────────
    random_seed: int = 42
    num_workers: int = 4

    # ── Distributed training ─────────────────────────────────────
    distributed: bool = True            # enable DDP
    dist_backend: str = "nccl"          # "nccl" for GPU, "gloo" for CPU
    find_unused_parameters: bool = False
    sync_batchnorm: bool = False        # convert BatchNorm → SyncBatchNorm
    # Per-GPU batch size.  Effective batch = batch_size × world_size.
    # Set via environment when using torchrun / PBS:
    #   MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
    master_port: str = "29500"

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def total_input_channels(self) -> int:
        """Input history + optional grid features."""
        extra = 2 if self.add_grid_features else 0
        return self.input_history + extra


@dataclass
class ModelInferenceConfig:
    """Parameters for Phase 3: Model Inference."""

    model_path: str = "checkpoints/best_model.pt"
    input_data_path: str = "data/processed/sst_standardised.nc"
    predictions_output_path: str = "results/predictions.nc"
    batch_size: int = 64

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ResultsAnalysisConfig:
    """Parameters for Phase 4: Results Analysis."""

    predictions_path: str = "results/predictions.nc"
    report_output_path: str = "results/report.txt"
    figures_output_dir: str = "results/figures/"
    metrics: List[str] = field(
        default_factory=lambda: ["mse", "rmse", "mae", "r2"]
    )
    generate_plots: bool = True


@dataclass
class ProjectConfig:
    """Top-level config aggregating all phase configs."""

    project_name: str = "oisst_regression"
    data_preparation: DataPreparationConfig = field(default_factory=DataPreparationConfig)
    model_training: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    model_inference: ModelInferenceConfig = field(default_factory=ModelInferenceConfig)
    results_analysis: ResultsAnalysisConfig = field(default_factory=ResultsAnalysisConfig)


# ── Global instance ──────────────────────────────────────────────────
# Import this from any phase module:
#     from config import cfg
cfg = ProjectConfig()
