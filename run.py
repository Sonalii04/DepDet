#!/usr/bin/env python3
"""
Depression Detection & Longitudinal Progression Framework
=========================================================
Unified Command-Line Runner (run.py)

This script provides a single, intuitive entry point for:
  - Running automated unit tests across all stages (50 tests)
  - Executing Stage 5 Policy A & B benchmarks, calibration, and threshold optimization
  - Executing Stage 4 longitudinal sequence GRU progression experiments
  - Executing Stage 3 tabular benchmark grid
  - Running data leakage and subject-isolation audit checks
  - Launching the standalone interactive Clinical Decision Support Dashboard

Usage examples:
  python run.py --test          # Run all 50 unit tests across 3 test suites
  python run.py --stage 5       # Run Stage 5 enhanced pipeline (benchmarks, SHAP, dashboard)
  python run.py --stage 4       # Run Stage 4 longitudinal GRU progression experiments
  python run.py --stage 3       # Run Stage 3 baseline and tabular benchmarks
  python run.py --dashboard     # Launch the interactive HTML5 decision support dashboard
  python run.py --audit         # Run automated data leakage and participant isolation checks
"""

import sys
import os
import argparse
import unittest
import webbrowser
from pathlib import Path

# Configure paths
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "project"
SRC_DIR = PROJECT_DIR / "src"

for p in (str(ROOT_DIR), str(PROJECT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_unit_tests():
    """Run all 50 automated unit tests across the 3 test suites."""
    print("=" * 80)
    print("RUNNING ALL AUTOMATED UNIT TESTS (Stages 1-5)")
    print("=" * 80)

    test_files = [
        "test_pipeline_and_leakage.py",
        "test_stage4_progression.py",
        "test_stage5_enhancements.py",
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for tf in test_files:
        test_path = SRC_DIR / tf
        if not test_path.exists():
            print(f"[ERROR] Test file not found: {test_path}")
            return 1
        suite.addTests(loader.discover(start_dir=str(SRC_DIR), pattern=tf))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"SUCCESS: All {result.testsRun} unit tests passed without errors or failures!")
        print("=" * 80)
        return 0
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests.")
        print("=" * 80)
        return 1


def run_leakage_audit():
    """Run automated data leakage and participant isolation checks."""
    print("=" * 80)
    print("RUNNING PARTICIPANT ISOLATION & DATA LEAKAGE AUDIT")
    print("=" * 80)
    from src.data_loader import load_raw_dataset, inspect_modalities_and_features
    from src.splitting import perform_subject_level_split
    from src.data_leakage_checks import run_all_data_leakage_checks

    parquet_path = ROOT_DIR / "anon_processed_df_parquet"
    if not parquet_path.exists():
        print(f"[ERROR] Dataset not found at: {parquet_path}")
        return 1

    df = load_raw_dataset(str(parquet_path))
    f_cols, _, _, _, _ = inspect_modalities_and_features(df)
    train_df, val_df, test_df, split_info = perform_subject_level_split(df, seed=42)

    passed = run_all_data_leakage_checks(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=f_cols,
        split_info=split_info
    )
    return 0 if passed else 1


def run_stage3():
    """Run Stage 3 tabular benchmark grid."""
    print("=" * 80)
    print("RUNNING STAGE 3 BENCHMARK GRID")
    print("=" * 80)
    from src.run_stage3_benchmark import run_stage3_benchmarks
    run_stage3_benchmarks()
    return 0


def run_stage4():
    """Run Stage 4 longitudinal progression experiments."""
    print("=" * 80)
    print("RUNNING STAGE 4 LONGITUDINAL PROGRESSION PIPELINE")
    print("=" * 80)
    from src.run_stage4_experiments import run_stage4_pipeline
    run_stage4_pipeline()
    return 0


def run_stage5():
    """Run Stage 5 master pipeline (Policy A/B, Calibration, Tuning, TreeSHAP, Dashboard)."""
    print("=" * 80)
    print("RUNNING STAGE 5 CONTROLLED ENHANCEMENT PIPELINE")
    print("=" * 80)
    from src.run_stage5_pipeline import run_stage5_master_pipeline
    run_stage5_master_pipeline()
    return 0


def open_dashboard():
    """Open or display path to the interactive HTML5 dashboard."""
    dash_path = PROJECT_DIR / "results" / "stage5_model_enhancements" / "clinical_decision_dashboard.html"
    print("=" * 80)
    print("CLINICAL DECISION SUPPORT DASHBOARD")
    print("=" * 80)
    if not dash_path.exists():
        print(f"[WARNING] Dashboard not generated yet at: {dash_path}")
        print("Run 'python run.py --stage 5' to generate it first.")
        return 1

    file_uri = dash_path.as_uri()
    print(f"Dashboard File: {dash_path}")
    print(f"Browser URI:    {file_uri}")
    print("\nAttempting to open in default web browser...")
    try:
        webbrowser.open(file_uri)
        print("Dashboard opened successfully!")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open this file manually in any modern browser: {dash_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Master CLI Runner for Explainable Longitudinal Depression Severity Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --test            Run all 50 unit tests across 3 suites
  python run.py --stage 5         Run Stage 5 Policy A/B benchmarks & enhancements
  python run.py --stage 4         Run Stage 4 longitudinal sequence GRU progression
  python run.py --stage 3         Run Stage 3 tabular benchmark grid
  python run.py --dashboard       Launch the interactive Clinical Dashboard
  python run.py --audit           Verify zero-leakage participant partitioning
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Run all 50 automated unit tests")
    group.add_argument("--stage", type=int, choices=[3, 4, 5], help="Execute pipeline for Stage 3, 4, or 5")
    group.add_argument("--pipeline", action="store_true", help="Execute complete Stage 5 pipeline (alias for --stage 5)")
    group.add_argument("--dashboard", action="store_true", help="Open the interactive HTML5 decision support dashboard")
    group.add_argument("--audit", action="store_true", help="Run automated data leakage and participant split checks")

    args = parser.parse_args()

    if args.test:
        sys.exit(run_unit_tests())
    elif args.stage == 3:
        sys.exit(run_stage3())
    elif args.stage == 4:
        sys.exit(run_stage4())
    elif args.stage == 5 or args.pipeline:
        sys.exit(run_stage5())
    elif args.dashboard:
        sys.exit(open_dashboard())
    elif args.audit:
        sys.exit(run_leakage_audit())


if __name__ == "__main__":
    main()
