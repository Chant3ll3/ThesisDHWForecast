"""
config.py — Single source of truth for all project parameters.

All four phases (data preparation, model training, model inference,
results analysis) read their settings from the dataclasses defined here.

RESOLUTION MACRO
----------------
Change the two lines below to update every resolution-dependent path
in the entire config simultaneously:

    LAT_SIZE = 128
    LON_SIZE = 256

All output directories and file paths that depend on resolution are
built from RESOLUTION_TAG = f"{LAT_SIZE}x{LON_SIZE}" automatically.
No other lines need to be edited when switching resolution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUTION MACRO — edit only these two lines to change resolution everywhere
# ══════════════════════════════════════════════════════════════════════════════
LAT_SIZE = 64      # number of latitude grid points
LON_SIZE = 128     # number of longitude grid points

# Derived tag used in all resolution-dependent paths, e.g. "128x256"
_RES = f"{LAT_SIZE}x{LON_SIZE}"
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataPreparationConfig:
    """Parameters for Phase 1: Data Preparation."""

    # ── Input ────────────────────────────────────────────────────
    input_dir: str = "/g/data/ia39/aus-ref-clim-data-nci/oisst/data/yearly/"
    file_pattern: str = "oisst-avhrr-v02r01_{year}.nc"
    year_range: Tuple[int, int] = (1981, 2024)
    variable: str = "sst"

    # ── Detrending ───────────────────────────────────────────────
    detrend_method: str = "global_linear"

    # ── Standardisation ──────────────────────────────────────────
    standardise_method: str = "global"

    # ── Land / missing-value fill ────────────────────────────────
    fill_method: str = "diffusion"
    fill_diffusion_iterations: int = 500
    fill_fallback_value: float = 0.0

    # ── Spatial downscaling ───────────────────────────────────────
    downscale: bool = True
    # Driven by the LAT_SIZE / LON_SIZE macro above.
    # Change the macro — these update automatically.
    target_lat_size: int = field(default_factory=lambda: LAT_SIZE)
    target_lon_size: int = field(default_factory=lambda: LON_SIZE)

    # ── Temporal averaging (applied after downscaling) ───────────
    # "none"    — keep daily resolution (default)
    # "weekly"  — 7-day (ISO week) means
    # "monthly" — calendar-month means
    temporal_average: str = "none"

    # ── Region subsetting (applied after fill) ───────────────────
    subset_region: bool = False
    lat_bounds: Optional[Tuple[float, float]] = None
    lon_bounds: Optional[Tuple[float, float]] = None

    # ── Diagnostics ──────────────────────────────────────────────
    diagnostic_year: int = 2020
    diagnostic_pdf: str = field(
        default_factory=lambda:
            f"diagnostics/data_preparation_diagnostics_{_RES}.pdf"
    )

    # ── Output ───────────────────────────────────────────────────
    # All paths include the resolution tag so outputs from different
    # resolution runs never overwrite each other.
    output_dir: str = field(
        default_factory=lambda: f"data/processed_{_RES}/"
    )
    save_intermediates: bool = True
    downscaled_file: str          = "sst_downscaled.nc"
    temporally_averaged_file: str = "sst_temporally_averaged.nc"
    detrended_file: str           = "sst_detrended.nc"
    standardised_file: str        = "sst_standardised.nc"
    filled_file: str              = "sst_filled.nc"
    final_file: str               = "sst_final.nc"
    subset_file: str              = "sst_subset.nc"
    trend_params_file: str        = "trend_params.nc"
    standardise_params_file: str  = "std_params.nc"

    # ── Resume ───────────────────────────────────────────────────
    #   0 = run from scratch (default)
    #   1 = resume after load       4 = resume after detrend
    #   2 = resume after downscale  5 = resume after standardise
    #   3 = resume after temporal avg  6 = resume after fill
    resume_after_stage: int = 0

    # ── Performance ──────────────────────────────────────────────
    chunks: dict = field(
        default_factory=lambda: {"time": 366, "lat": 720, "lon": 720}
    )
    dask_scheduler: str       = "threads"
    dask_n_workers: int       = 8
    dask_threads_per_worker: int = 2
    dask_memory_limit: str    = "12GB"
    dask_num_threads: int     = 16


@dataclass
class ModelTrainingConfig:
    """Parameters for Phase 2: Model Training (SFNO for SST forecasting)."""

    # ── Select Model ─────────────────────────────────────────────
    model_name: str = "sfno"   # "sfno" or "convlstm"

    # ── SFNO Architecture ────────────────────────────────────────
    input_history: int = 10             # past time steps as input channels
    output_channels: int = 1            # predict single SST field at t+1
    embed_dim: int = 80                 # embedding / hidden channel width
    num_layers: int = 5                 # number of SFNO blocks
    num_modes_lat: int = 16             # spectral truncation in latitude
    num_modes_lon: int = 32             # spectral truncation in longitude
    mlp_ratio: float = 2.5             # MLP hidden dim = embed_dim * mlp_ratio
    activation: str = "gelu"            # "gelu", "relu", "silu"
    norm_type: str = "instance_norm"    # "layer_norm", "instance_norm", "none"
    dropout_rate: float = 0.05          # randomly turns off 20% of neurons, prevents overfitting
    use_residual: bool = True           # global skip from input to output
    spectral_bias: bool = True          # learnable bias in spectral domain

    # ── Positional / geographic encoding ─────────────────────────
    add_grid_features: bool = True      # adds 2 extra input channels (lat, lon)

    # ── Seasonal cycle encoding ──────────────────────────────────
    # Appends sin(2π · day_of_year/365.25) and cos(...) as extra
    # input channels. DISABLED by default — spatially uniform
    # encoding conflicts with spectral convolutions. Consider
    # injecting season via a separate MLP bias instead.
    add_seasonal_encoding: bool = False  # adds 2 extra input channels


    # ── Forecasting strategy ─────────────────────────────────────
    forecast_horizon: int = 90          # maximum lead time in steps
    # Autoregressive rollout: ramp gradually during training.
    # Start early so the LR is still high enough to adapt.
    initial_rollout_steps: int = 1
    max_rollout_steps: int = 3          # keep modest (not 15), reduce if OOM
    rollout_epoch_start: int = 2       # start ramping 
    rollout_epoch_end: int = 40        # reach max by epoch
    rollout_loss_decay: float = 0.95    # exponential weight per step

    # ── Data splitting ───────────────────────────────────────────
    train_years: Tuple[int, int] = (1982, 2015)  #training data
    val_years: Tuple[int, int] = (2016, 2019)    #validation data
    test_years: Tuple[int, int] = (2020, 2024)   #test data

    # ── Training hyperparameters ─────────────────────────────────
    learning_rate: float = 0.0002          # controls how big each update step
    min_learning_rate: float = 1e-06      # the smallest the learning rate can decrease to during training
    batch_size: int = 16                 # number of samples processed before updating the model, uses memory reduce if OOM
    num_epochs: int = 120                # number of times the model sees the entire dataset
    weight_decay: float = 1e-05           # penalises large weights to improve generalisation
    optimizer: str = "adamw"             # algorithm used to update model weights efficiently
    scheduler: str = "cosine"            # gradually reduced the learning rate in a smooth curve
    scheduler_patience: int = 10         # waits 10 epochs (one complete pass of the entire training dataset) before adjusting learning if no imporovement
    scheduler_factor: float = 0.5        # reduces the learning rate by half when triggered (if applicable)
    warmup_epochs: int = 5               # slowly increases learning rate at the start to stabilise training
    gradient_clip_norm: float = 1.0      # limits gradient size to prevent unstable updates
    loss_function: str = "huber"           # measures prediction error using squared differences

    # ── Early stopping ───────────────────────────────────────────
    early_stopping_patience: int = 20        # stops training if no imporvement for 20 epochs
    early_stopping_metric: str = "val_loss"  # uses validation error to decide when to stop

    # ── Paths — resolution-tagged ─────────────────────────────────
    input_data_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/sst_final.nc"
    )
    checkpoint_dir: str = field(
        default_factory=lambda: f"checkpoints_{_RES}/"
    )
    best_model_path: str = field(
        default_factory=lambda: f"checkpoints_{_RES}/best_model.pt"
    )
    training_log_path: str = field(
        default_factory=lambda: f"results_{_RES}/training_log.csv"
    )

    # ── Reproducibility ──────────────────────────────────────────
    random_seed: int = 42  # ensures results are consistent across runs
    num_workers: int = 4   # number of cpus used to load data faster

    # ── Distributed training ─────────────────────────────────────
    distributed: bool = True                # enables training across multiple GPUs
    dist_backend: str = "nccl"              # uses NVIDIA's backend for fast GPU communication
    find_unused_parameters: bool = False    # assumes all model parameters are used
    sync_batchnorm: bool = False            # does not synchronise batch statistics across GPUs
    master_port: str = "29500"              # communication port used for coordinating GPUs

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def total_input_channels(self) -> int:
        """Input history + optional grid features + optional seasonal encoding."""
        extra = 0
        if self.add_grid_features:    extra += 2
        if self.add_seasonal_encoding: extra += 2
        return self.input_history + extra

    @property
    def seasonal_cycle_period(self) -> float:
        """Period of the seasonal cycle in days.
        Always 365.25 because day_of_year is always in actual days
        regardless of temporal resolution."""
        return 365.25


@dataclass
class ModelInferenceConfig:
    """Parameters for Phase 3: Model Inference."""

    # ── Model ────────────────────────────────────────────────────
    model_path: str = field(
        default_factory=lambda: f"checkpoints_{_RES}/best_model.pt"
    )

    # ── Data ─────────────────────────────────────────────────────
    input_data_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/sst_final.nc"
    )
    run_validation: bool = True
    run_test: bool       = True

    # ── Autoregressive rollout ───────────────────────────────────
    # Lead time and stride are in TIME STEPS (days/weeks/months
    # depending on temporal_average setting in Phase 1).
    forecast_lead_steps: int = 90       # initially 14, can increase to 90
    chain_stride_steps: int = 90        # initially 14, can increase to 90

    # ── Output ───────────────────────────────────────────────────
    output_dir: str = field(
        default_factory=lambda: f"results_{_RES}/"
    )
    val_predictions_file: str  = "predictions_val.nc"
    test_predictions_file: str = "predictions_test.nc"

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ResultsAnalysisConfig:
    """Parameters for Phase 4: Results Analysis."""

    # ── Input ────────────────────────────────────────────────────
    val_predictions_path: str = field(
        default_factory=lambda: f"results_{_RES}/predictions_val.nc"
    )
    test_predictions_path: str = field(
        default_factory=lambda: f"results_{_RES}/predictions_test.nc"
    )
    processed_data_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/sst_final.nc"
    )
    analyse_validation: bool = True
    analyse_test: bool       = True

    # ── Standardisation params (for converting back to °C) ──────
    std_params_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/std_params.nc"
    )
    trend_params_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/trend_params.nc"
    )

    # ── Baselines ────────────────────────────────────────────────
    compute_persistence: bool  = True
    compute_climatology: bool  = True

    # ── ACC climatology ──────────────────────────────────────────
    # Daily (day-of-year) climatology for ACC anomaly computation.
    # Computed from the processed data over a 30-year baseline.
    acc_clim_years: Tuple[int, int] = (1991, 2020)
    acc_clim_smoothing_window: int  = 5 # rolling mean window (days)

    # ── Sample forecast chains to plot ───────────────────────────
    n_sample_chains: int = 9            # 3×3 grid of example chains

    # ── Spatial analysis ─────────────────────────────────────────
    # Lead steps for spatial RMSE/ACC maps and ENSO scatter (3×3 grid)
    spatial_lead_steps: List[int] = field(
        default_factory=lambda: [1, 3, 7, 14, 21, 30, 45, 60, 90]
    )

    # ── ENSO 3.4 analysis ────────────────────────────────────────
    compute_enso34: bool                    = True
    enso34_lat_bounds: Tuple[float, float]  = (-5.0, 5.0)
    enso34_lon_bounds: Tuple[float, float]  = (190.0, 240.0)
    enso34_clim_years: Tuple[int, int]      = (1991, 2020)
    enso34_n_sample_chains: int             = 9 # 3×3 grid of example chains

    compute_dhw: bool                      = True
    dhw_baseline_years: Tuple[int, int]    = (1985, 2012)
    dhw_output_val: str                    = "dhw_val.nc"
    dhw_output_test: str                   = "dhw_test.nc"

    # ── Output ───────────────────────────────────────────────────
    output_dir: str = field(
        default_factory=lambda: f"results_{_RES}/"
    )
    report_file: str = field(
        default_factory=lambda: f"results_{_RES}/analysis_report.txt"
    )
    metrics_csv: str = field(
        default_factory=lambda: f"results_{_RES}/metrics_by_lead_step.csv"
    )
    figures_dir: str = field(
        default_factory=lambda: f"results_{_RES}/figures/"
    )
    analysis_pdf: str = field(
        default_factory=lambda: f"results_{_RES}/forecast_analysis.pdf"
    )
    generate_plots: bool = True


@dataclass
class ConcatChainsConfig:
    """
    Parameters for ConcatChainsUpdated.py (Phase 3.5).

    Reverses preprocessing transformations to recover real SST in Celsius,
    concatenates actual SST into a continuous timeline, and saves both
    files ready for DHW_calculation.py.
    """

    predictions_dir: str = field(
        default_factory=lambda:
            f"/scratch/um09/xl5758/DHW-project/results_{_RES}"
    )
    predictions_file_template: str = "predictions_{split}.nc"

    std_params_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/std_params.nc"
    )
    trend_params_path: str = field(
        default_factory=lambda: f"data/processed_{_RES}/trend_params.nc"
    )

    output_dir: str = field(
        default_factory=lambda:
            f"/scratch/um09/xl5758/DHW-project/results_{_RES}"
    )
    actual_file_template: str    = "sst_actual_{split}_continuous.nc"
    predicted_file_template: str = "sst_predicted_{split}_chains.nc"

    splits: List[str] = field(default_factory=lambda: ["test", "val"])

    # Lon correction: stored span < threshold * expected span triggers fix.
    # Set to 0.0 to disable.
    lon_correction_threshold: float = 0.6


@dataclass
class CalcDHWConfig:
    """
    Parameters for Calc_DHW02.py (Phase 5.2).

    Uses the NOAA CRW blended sliding window:
      Lead L: (84-L) days actual SST + L days predicted SST
    Error at lead 1 is near zero; grows naturally with lead time.
    Also computes a rigorous persistence baseline (init-day SST held constant).
    """

    # NOAA CRW ct5km climatology — sst_clim_mmm variable.
    mmm_path: str = "/home/272/xl5758/ct5km_climatology_v3.1.nc"

    # Input directory (ConcatChainsUpdated.py outputs)
    results_dir: str = field(
        default_factory=lambda:
            f"/scratch/um09/xl5758/DHW-project/results_{_RES}"
    )

    predicted_file_template: str = "sst_predicted_{split}_chains.nc"
    actual_file_template: str    = "sst_actual_{split}_continuous.nc"

    # DHW products written here (separate from raw SST files)
    dhw_output_dir: str = field(
        default_factory=lambda:
            f"/scratch/um09/xl5758/DHW-project/dhw_output_{_RES}"
    )
    output_file_template: str = "dhw_products_{split}.nc"

    run_test: bool       = True
    run_validation: bool = True

    window_days: int         = 84    # NOAA CRW standard: 84 days (12 weeks)
    hotspot_threshold: float = 1.0   # NOAA CRW standard: 1.0 degC


@dataclass
class DHWPlotsConfig:
    """
    Parameters for dhw_plots.py (Phase 5.3).

    Controls all output paths, plot appearance, and which forecast chains
    are shown.  Edit only this block — do not hardcode values in dhw_plots.py.
    """

    dhw_dir: str = field(
        default_factory=lambda:
            f"/scratch/um09/xl5758/DHW-project/dhw_output_{_RES}"
    )
    output_pdf: str = "dhw_analysis_combined.pdf"
    output_csv: str = "dhw_metrics_combined.csv"

    spatial_lead_steps: List[int] = field(
        default_factory=lambda: [1, 7, 14, 30, 60, 90]
    )

    n_sample_chains: int = 6

    # Specific init dates to plot.  Set to [] for auto (n_sample_chains).
    plot_init_dates: List[str] = field(
        default_factory=lambda: [
            "2016-05-20", "2016-10-15",
            "2017-02-10", "2017-07-09",
            "2018-03-15", "2018-06-20", "2019-09-27",
            "2020-06-10", "2021-01-01", "2021-04-27",
            "2021-09-09", "2022-11-15", "2023-08-20",
        ]
    )

    dhw_vmax: float      = 20.0   # DHW colour scale ceiling (C-weeks)
    dhw_diff_vmax: float =  8.0   # error colour scale symmetric cap
    tropic_lat: float    = 30.0   # tropical band for area-under-stress plots

    # ── Colour maps ───────────────────────────────────────────────
    # String → named Matplotlib cmap.  List → smooth custom gradient.

    # NOAA-inspired DHW scale (light blue → purple → yellow → red → dark):
    dhw_cmap: object = field(
        default_factory=lambda: [
            "#CCe6F0", "#441C94", "#6F47BF", "#815CCC", "#A383E6",
            "#EDE61F", "#EDCB1F", "#EDAF1F", "#ED901F", "#ED1F1F",
            "#A11212", "#731616", "#520C0C", "#D67454", "#A1573F",
            "#804532", "#592F22", "#ED55E6", "#BF43BA", "#942F90",
            "#6E1F6B", "#380A36",
        ]
    )

    # Blue → white → red diverging error scale:
    error_cmap: object = field(
        default_factory=lambda: [
            "#053061", "#2166AC", "#92C5DE",
            "#F7F7F7",
            "#F4A582", "#D6604D", "#67001F",
        ]
    )

    rmse_cmap: str = "YlOrRd"   # spatial RMSE maps
    bias_cmap: str = "RdBu_r"   # spatial bias maps

    # ── Bleaching alert thresholds ────────────────────────────────
    bleaching_thresholds: List[Tuple] = field(
        default_factory=lambda: [
            (0,   4,  "No Stress",       "#27AE60"),
            (4,   8,  "Warning",         "#F39C12"),
            (8,  12,  "Alert Level 1",   "#E74C3C"),
            (12, 16,  "Alert Level 2",   "#C0392B"),
            (16, 20,  "Alert Level 3",   "#8E44AD"),
            (20, 999, "Alert Level 4-5", "#4A235A"),
        ]
    )

    # ── Great Barrier Reef bounding box ──────────────────────────
    gbr_lat_bounds: Tuple[float, float] = (-25.0, -10.0)
    gbr_lon_bounds: Tuple[float, float] = (142.0, 157.0)

    # ── Major coral reef systems ──────────────────────────────────
    # Each entry: name, lat (min, max), lon (min, max in 0-360), colour.
    # Boxes are drawn on every map and listed in the legend.
    # Comment out entries to hide them.
    reef_boxes: List[dict] = field(
        default_factory=lambda: [
            {
                "name":   "Great Barrier Reef",
                "lat":    (-25.0, -10.0),
                "lon":    (142.0, 157.0),
                "colour": "#FFD700",    # gold
            },
            # {
            #     "name":   "Red Sea Coral Reef",
            #     "lat":    (12.0, 28.0),
            #     "lon":    (32.0, 44.0),
            #     "colour": "#FF4500",    # orange-red
            # },
            # {
            #     "name":   "New Caledonia Barrier Reef",
            #     "lat":    (-23.0, -18.0),
            #     "lon":    (163.0, 168.0),
            #     "colour": "#00CED1",
            # },
            # {
            #     "name":   "Mesoamerican Barrier Reef",
            #     "lat":    (15.0, 22.0),
            #     "lon":    (273.0, 279.0),
            #     "colour": "#32CD32",
            # },
            # {
            #     "name":   "Florida Reef",
            #     "lat":    (24.0, 27.0),
            #     "lon":    (279.0, 281.0),
            #     "colour": "#FF69B4",
            # },
            # {
            #     "name":   "Andros Coral Reef",
            #     "lat":    (23.0, 26.0),
            #     "lon":    (282.0, 284.0),
            #     "colour": "#DA70D6",
            # },
            # {
            #     "name":   "Saya de Malha Bank",
            #     "lat":    (-17.0, -10.0),
            #     "lon":    (60.0, 64.0),
            #     "colour": "#FF8C00",
            # },
            # {
            #     "name":   "Coral Triangle",
            #     "lat":    (-12.0, 8.0),
            #     "lon":    (115.0, 135.0),
            #     "colour": "#00BFFF",
            # },
        ]
    )


@dataclass
class ProjectConfig:
    """Top-level config aggregating all phase configs."""

    project_name: str = "oisst_regression"
    data_preparation: DataPreparationConfig = field(
        default_factory=DataPreparationConfig)
    model_training: ModelTrainingConfig = field(
        default_factory=ModelTrainingConfig)
    model_inference: ModelInferenceConfig = field(
        default_factory=ModelInferenceConfig)
    results_analysis: ResultsAnalysisConfig = field(
        default_factory=ResultsAnalysisConfig)
    concat_chains: ConcatChainsConfig = field(
        default_factory=ConcatChainsConfig)
    calc_dhw: CalcDHWConfig = field(default_factory=CalcDHWConfig)
    dhw_plots: DHWPlotsConfig = field(default_factory=DHWPlotsConfig)


# ── Global instance ───────────────────────────────────────────────────────────
cfg = ProjectConfig()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Resolution: {LAT_SIZE} lat x {LON_SIZE} lon  (tag: {_RES})")
    print(f"  data_preparation.output_dir      = {cfg.data_preparation.output_dir}")
    print(f"  model_training.input_data_path   = {cfg.model_training.input_data_path}")
    print(f"  model_training.checkpoint_dir    = {cfg.model_training.checkpoint_dir}")
    print(f"  model_inference.output_dir       = {cfg.model_inference.output_dir}")
    print(f"  results_analysis.output_dir      = {cfg.results_analysis.output_dir}")
    print(f"  concat_chains.output_dir         = {cfg.concat_chains.output_dir}")
    print(f"  calc_dhw.results_dir             = {cfg.calc_dhw.results_dir}")
    print(f"  calc_dhw.dhw_output_dir          = {cfg.calc_dhw.dhw_output_dir}")
    print(f"  dhw_plots.dhw_dir                = {cfg.dhw_plots.dhw_dir}")
    print(f"  data_preparation.target_lat_size = {cfg.data_preparation.target_lat_size}")
    print(f"  data_preparation.target_lon_size = {cfg.data_preparation.target_lon_size}")