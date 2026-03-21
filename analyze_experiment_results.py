"""
分析実験結果：計算精度ドロップと状態数削減
Analysis of Experiment Results: Calculate accuracy improvements and state reduction
"""

import os
import pandas as pd
import numpy as np
from typing import Dict

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RESULT_DIRS = [
    os.path.join(PROJECT_ROOT, "test_result", "baseline_experiment_threshold_0.8"),
    os.path.join(PROJECT_ROOT, "test_result", "baseline_experiment_threshold_0.9"),
]


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """Load results CSV file."""
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to load {csv_path}: {e}")
        return None


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate improvements and reductions from experimental results.
    
    Returns dict with structure:
    {
        'summary': DataFrame with aggregated metrics per method,
        'by_language': Dict[language] -> DataFrame with per-language metrics,
        'overall': Dict with overall statistics
    }
    """
    if df is None or len(df) == 0:
        return None
    
    results = {}
    
    # Per-method and per-language analysis
    analysis_data = []
    
    for _, row in df.iterrows():
        lang = row['language']
        method = row['method']
        
        # 訓練準確度の改善
        initial_train = float(row['initial_train_acc'])
        final_train = float(row['final_train_acc'])
        train_improvement = final_train - initial_train
        
        # 驗證準確度の改善
        initial_val = float(row['initial_validation_acc'])
        final_val = float(row['final_validation_acc'])
        val_improvement = final_val - initial_val
        
        # 狀態數削減 (handle NaN values)
        init_states = int(row['init_states'])
        final_states_val = row['states']
        if pd.isna(final_states_val):
            # Beam 方法没有 states 值
            final_states = None
            states_reduction = None
            states_reduction_pct = None
        else:
            final_states = int(final_states_val)
            states_reduction = final_states - init_states  # 後減前
            states_reduction_pct = (states_reduction / init_states * 100) if init_states > 0 else 0
        
        # 実行時間
        time_s = float(row['time_s'])
        
        # 成功フラグ
        success = int(row['success'])
        
        analysis_data.append({
            'language': lang,
            'method': method,
            'initial_train_acc': initial_train,
            'final_train_acc': final_train,
            'train_acc_improvement': train_improvement,
            'initial_validation_acc': initial_val,
            'final_validation_acc': final_val,
            'val_acc_improvement': val_improvement,
            'init_states': init_states,
            'final_states': final_states,
            'states_reduction': states_reduction,
            'states_reduction_pct': states_reduction_pct,
            'time_s': time_s,
            'success': success,
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    return {
        'full_data': analysis_df,
        'by_language': analysis_df.groupby('language'),
        'by_method': analysis_df.groupby('method'),
    }


def print_summary(threshold: float, analysis: Dict) -> None:
    """Print formatted summary of analysis."""
    if analysis is None:
        print(f"\n  [accuracy_threshold={threshold}] No data to analyze\n")
        return
    
    df = analysis['full_data']
    
    print(f"\n{'='*110}")
    print(f"  EXPERIMENT ANALYSIS (accuracy_threshold={threshold})")
    print(f"{'='*110}")
    
    # Summary by language
    print(f"\n  ┌─ Results by Language")
    print(f"  │")
    
    for lang in sorted(df['language'].unique()):
        lang_df = df[df['language'] == lang]
        print(f"  │  Language: {lang}")
        print(f"  │  Initial states: {lang_df['init_states'].iloc[0]:>3d}")
        print(f"  │")
        print(f"  │  {'Method':<10s} {'Train Initial':>13s} {'Train Final':>13s} {'Train Δ':>10s} {'Valid Initial':>13s} {'Valid Final':>13s} {'Valid Δ':>10s} {'States':>8s} {'States Δ':>10s} {'Time':>8s}")
        print(f"  │  {'-'*120}")
        
        for method in ['beam', 'sa', 'ga', 'pso']:
            method_rows = lang_df[lang_df['method'] == method]
            if len(method_rows) > 0:
                row = method_rows.iloc[0]
                states_str = f"{row['final_states']:.0f}" if (row['final_states'] is not None and not pd.isna(row['final_states'])) else "N/A"
                states_delta_str = f"{row['states_reduction']:.0f}" if (row['states_reduction'] is not None and not pd.isna(row['states_reduction'])) else "N/A"
                print(f"  │  {method:<10s} {row['initial_train_acc']:>13.4f} {row['final_train_acc']:>13.4f} {row['train_acc_improvement']:>+10.4f} "
                      f"{row['initial_validation_acc']:>13.4f} {row['final_validation_acc']:>13.4f} {row['val_acc_improvement']:>+10.4f} "
                      f"{states_str:>8s} {states_delta_str:>10s} {row['time_s']:>8.1f}s")
        print(f"  │")
    
    print(f"  └─")
    
    # Per-language detail - DISABLED
    # print(f"\n  ┌─ Per-Language Details")
    # print(f"  │")
    
    # for lang in sorted(df['language'].unique()):
    #     lang_df = df[df['language'] == lang]
    #     print(f"  │  Language: {lang}")
    #     print(f"  │  Initial states: {lang_df['init_states'].iloc[0]:>3d}")
    #     print(f"  │")
    #     print(f"  │  {'Method':<12s} {'Train Δ':>10s} {'Valid Δ':>10s} {'States':>8s} {'States Δ':>10s} {'Time':>8s}")
    #     print(f"  │  {'-'*70}")
    #
    #     for method in ['beam', 'sa', 'ga', 'pso']:
    #         method_rows = lang_df[lang_df['method'] == method]
    #         if len(method_rows) > 0:
    #             row = method_rows.iloc[0]
    #             states_str = f"{row['final_states']:.0f}" if (row['final_states'] is not None and not pd.isna(row['final_states'])) else "N/A"
    #             states_delta_str = f"{row['states_reduction']:.0f}" if (row['states_reduction'] is not None and not pd.isna(row['states_reduction'])) else "N/A"
    #             print(f"  │  {method:<12s} {row['train_acc_improvement']:>+10.4f} "
    #                   f"{row['val_acc_improvement']:>+10.4f} {states_str:>8s} "
    #                   f"{states_delta_str:>10s} {row['time_s']:>8.1f}s")
    #     print(f"  │")
    
    # print(f"  └─")
    print()


def save_detailed_csv(analysis: Dict, output_path: str) -> None:
    """Save detailed analysis to CSV file."""
    if analysis is None:
        return
    
    df = analysis['full_data'].copy()
    
    # Replace None values with empty strings for CSV output
    for col in ['final_states', 'states_reduction', 'states_reduction_pct']:
        df[col] = df[col].apply(lambda x: '' if (pd.isna(x) or x is None) else x)
    
    # Select relevant columns
    output_columns = [
        'language', 'method',
        'initial_train_acc', 'final_train_acc', 'train_acc_improvement',
        'initial_validation_acc', 'final_validation_acc', 'val_acc_improvement',
        'init_states', 'final_states', 'states_reduction', 'states_reduction_pct',
        'time_s', 'success'
    ]
    
    output_df = df[output_columns].copy()
    output_df = output_df.sort_values(['language', 'method'])
    
    output_df.to_csv(output_path, index=False)
    print(f"  Detailed analysis saved → {output_path}")


if __name__ == "__main__":
    print("\n" + "="*110)
    print("  ANALYZING BASELINE EXPERIMENT RESULTS")
    print("="*110)
    
    all_results = {}
    
    for result_dir in RESULT_DIRS:
        if not os.path.exists(result_dir):
            print(f"\n  [SKIP] Directory not found: {result_dir}")
            continue
        
        csv_file = os.path.join(result_dir, "results.csv")
        
        # Extract threshold from directory name
        threshold_str = result_dir.split("_")[-1]
        try:
            threshold = float(threshold_str)
        except:
            threshold = "unknown"
        
        print(f"\n  Loading results from: {threshold_str}")
        
        df = load_results_csv(csv_file)
        if df is not None:
            print(f"  ✓ Loaded {len(df)} rows")
            
            # Analyze
            analysis = calculate_metrics(df)
            
            # Print summary
            print_summary(threshold, analysis)
            
            # Save detailed results
            output_csv = os.path.join(result_dir, "analysis.csv")
            save_detailed_csv(analysis, output_csv)
            
            all_results[threshold] = analysis
    
    # Comparative analysis
    if len(all_results) > 1:
        print("\n" + "="*110)
        print("  COMPARATIVE ANALYSIS (0.8 vs 0.9)")
        print("="*110)
        
        thresholds = sorted(all_results.keys())
        if len(thresholds) == 2:
            df0_8 = all_results[0.8]['full_data']
            df0_9 = all_results[0.9]['full_data']
            
            print(f"\n  ┌─ Impact of Accuracy Threshold on Method Performance")
            print(f"  │")
            print(f"  │  Training Accuracy Improvement (mean across languages):")
            print(f"  │  {'Method':<10s} {'threshold=0.8':>15s} {'threshold=0.9':>15s} {'Δ':>15s}")
            print(f"  │  {'-'*55}")
            
            for method in ['beam', 'sa', 'ga', 'pso']:
                val0_8 = df0_8[df0_8['method'] == method]['train_acc_improvement'].mean()
                val0_9 = df0_9[df0_9['method'] == method]['train_acc_improvement'].mean()
                delta = val0_9 - val0_8
                print(f"  │  {method:<10s} {val0_8:>15.4f} {val0_9:>15.4f} {delta:>+15.4f}")
            
            print(f"  │")
            print(f"  │  State Reduction % (mean across languages):")
            print(f"  │  {'Method':<10s} {'threshold=0.8':>15s} {'threshold=0.9':>15s} {'Δ':>15s}")
            print(f"  │  {'-'*55}")
            
            for method in ['beam', 'sa', 'ga', 'pso']:
                val0_8_data = df0_8[df0_8['method'] == method]['states_reduction_pct'].dropna()
                val0_9_data = df0_9[df0_9['method'] == method]['states_reduction_pct'].dropna()
                
                val0_8 = val0_8_data.mean() if len(val0_8_data) > 0 else float('nan')
                val0_9 = val0_9_data.mean() if len(val0_9_data) > 0 else float('nan')
                
                if not np.isnan(val0_8) and not np.isnan(val0_9):
                    delta = val0_9 - val0_8
                    print(f"  │  {method:<10s} {val0_8:>14.2f}% {val0_9:>14.2f}% {delta:>+14.2f}%")
                else:
                    print(f"  │  {method:<10s} {'N/A':>15s} {'N/A':>15s} {'N/A':>15s}")
            
            print(f"  │")
            print(f"  │  Execution Time (seconds, mean):")
            print(f"  │  {'Method':<10s} {'threshold=0.8':>15s} {'threshold=0.9':>15s} {'Δ':>15s}")
            print(f"  │  {'-'*55}")
            
            for method in ['beam', 'sa', 'ga', 'pso']:
                val0_8 = df0_8[df0_8['method'] == method]['time_s'].mean()
                val0_9 = df0_9[df0_9['method'] == method]['time_s'].mean()
                delta = val0_9 - val0_8
                print(f"  │  {method:<10s} {val0_8:>15.2f} {val0_9:>15.2f} {delta:>+15.2f}")
            
            print(f"  └─")
    
    print("\n  ✓ Analysis complete!\n")
