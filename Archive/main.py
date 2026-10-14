"""
main.py — Orchestrator for the OISST–SFNO pipeline.

Run individual phases or the full pipeline:

  python main.py                      # all phases sequentially
  python main.py --phase 1            # data preparation only
  python main.py --phase 2            # model training only
  python main.py --phase 3            # model inference only
  python main.py --phase 4            # results analysis only
  python main.py --phase 1 2          # phases 1 then 2

Resume Phase 1 after a completed stage:

  python main.py --phase 1 --resume-after 4   # resume after standardise
  python main.py --phase 1 --resume-after 3   # resume after detrend

  Stage numbers:  1=load  2=downscale  3=detrend  4=standardise  5=fill

For distributed training (Phase 2), launch with torchrun:

  torchrun --nproc_per_node=4 model_training.py
  torchrun --nproc_per_node=4 main.py --phase 2
"""

import argparse
import sys

from config import cfg


PHASE_MAP = {
    1: ("data_preparation", "Phase 1: Data Preparation"),
    2: ("model_training",   "Phase 2: Model Training"),
    3: ("model_inference",  "Phase 3: Model Inference"),
    4: ("results_analysis", "Phase 4: Results Analysis"),
}


def run_phase(phase_num: int):
    """Import and run the specified phase."""
    if phase_num not in PHASE_MAP:
        print(f"ERROR: Unknown phase {phase_num}. Valid phases: 1, 2, 3, 4")
        sys.exit(1)

    module_name, description = PHASE_MAP[phase_num]
    print(f"\n{'─' * 60}")
    print(f"  Running {description}")
    print(f"{'─' * 60}\n")

    # Dynamic import so unused phases don't need all dependencies loaded
   # import importlib
   # module = importlib.import_module(module_name)
   # module.run()


def main():
    parser = argparse.ArgumentParser(
        description="OISST–SFNO SST Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", "-p",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=None,
        help="Phase(s) to run: 1=DataPrep, 2=Training, 3=Inference, 4=Analysis. "
             "Default: run all in order.",
    )
    parser.add_argument(
        "--resume-after", "-r",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=None,
        help="(Phase 1 only) Resume after this completed stage: "
             "0=scratch, 1=load, 2=downscale, 3=detrend, 4=standardise, 5=fill.",
    )
    args = parser.parse_args()

    # Apply resume override to config
    if args.resume_after is not None:
        cfg.data_preparation.resume_after_stage = args.resume_after

    print(f"\n🚀 Project: {cfg.project_name}\n")

    phases = args.phase if args.phase else [1, 2, 3, 4]

    for p in phases:
        run_phase(p)

    print("\n" + "=" * 60)
    print(f"  FINISHED — Phases {phases}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
