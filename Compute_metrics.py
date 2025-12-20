import glob
import yaml
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def compute_aggressiveness_metrics(trajectory, dt=0.01):
    """
    Compute trajectory aggressiveness metrics
    
    IMPORTANT: Only EST data gets downsampled because the zeros are an
    implementation detail. RRT/RRT* data is NOT downsampled - each timestep
    represents an actual controller execution.
    
    Parameters:
    -----------
    trajectory : dict
        Trajectory data in RRT, RRT*, or EST format
    dt : float
        Integration time step (default: 0.01 s)
    
    Returns:
    --------
    dict : metrics or None if trajectory too short
    """
    
    # Initialize time_diffs as None
    time_diffs = None
    
    # Case 1: RRT* format - nested under iteration key
    if 'commands' not in trajectory and 'tau_list' not in trajectory:
        iteration_keys = [k for k in trajectory.keys() if 'iterations' in str(k).lower()]
        
        if len(iteration_keys) > 0:
            iter_key = iteration_keys[0]
            if 'commands' in trajectory[iter_key]:
                commands = np.array(trajectory[iter_key]['commands'])
                if len(commands) < 3:
                    return None
                tau = commands[:, 0]
                F = commands[:, 1]
                data_source = "RRT*"
            else:
                return None
        else:
            return None
    
    # Case 2: RRT format - direct commands
    elif 'commands' in trajectory:
        commands = np.array(trajectory['commands'])
        if len(commands) < 3:
            return None
        tau = commands[:, 0]
        F = commands[:, 1]
        data_source = "RRT"
    
    # Case 3: EST format - tau_list and f_list
    elif 'tau_list' in trajectory and 'f_list' in trajectory:
        tau = np.array(trajectory['tau_list'])
        F = np.array(trajectory['f_list'])
        
        if len(tau) < 3 or len(F) < 3:
            return None
        
        if len(tau) != len(F):
            return None
        
        data_source = "EST"
        original_length = len(tau)
        
        # ALWAYS DOWNSAMPLE EST
        keep_indices = [0]  # Always keep first
        
        for i in range(1, len(tau)):
            tau_changed = np.abs(tau[i] - tau[i-1]) > 1e-10
            F_changed = np.abs(F[i] - F[i-1]) > 1e-10
            
            if tau_changed or F_changed:
                keep_indices.append(i)
        
        # Always keep last point if not already there
        if keep_indices[-1] != len(tau) - 1:
            keep_indices.append(len(tau) - 1)
        
        # Extract the commands at change points
        tau = tau[keep_indices]
        F = F[keep_indices]
        
        # Compute actual time differences between commands
        time_diffs = np.diff(keep_indices) * dt
        
        compression_ratio = original_length / len(tau)
        print(f"  EST downsample: {original_length} → {len(tau)} ({compression_ratio:.1f}x)")
        print(f"  Mean time between commands: {np.mean(time_diffs):.3f} s")
    
    else:
        return None
    
    n_steps = len(tau)
    
    metrics = {}
    
    # === BASIC STATISTICS ===
    metrics['force_mean'] = np.mean(F)
    metrics['force_std'] = np.std(F)
    metrics['force_variance'] = np.var(F)
    metrics['force_min'] = np.min(F)
    metrics['force_max'] = np.max(F)
    
    metrics['torque_mean'] = np.mean(tau)
    metrics['torque_std'] = np.std(tau)
    metrics['torque_variance'] = np.var(tau)
    metrics['torque_min'] = np.min(tau)
    metrics['torque_max'] = np.max(tau)
    
    # === FIRST DERIVATIVE (Rate of change) ===
    if time_diffs is not None:
        # EST: use actual time between commands
        dF_dt = np.diff(F) / time_diffs
        dtau_dt = np.diff(tau) / time_diffs
    else:
        # RRT/RRT*: use constant dt
        dF_dt = np.diff(F) / dt
        dtau_dt = np.diff(tau) / dt
    
    metrics['max_force_rate'] = np.max(np.abs(dF_dt))
    metrics['mean_force_rate'] = np.mean(np.abs(dF_dt))
    metrics['max_torque_rate'] = np.max(np.abs(dtau_dt))
    metrics['mean_torque_rate'] = np.mean(np.abs(dtau_dt))
    
    # === SECOND DERIVATIVE (Jerk - smoothness) ===
    if time_diffs is not None and len(time_diffs) > 1:
        # EST: time between rate measurements
        dt_for_jerk = (time_diffs[:-1] + time_diffs[1:]) / 2
        d2F_dt2 = np.diff(dF_dt) / dt_for_jerk
        d2tau_dt2 = np.diff(dtau_dt) / dt_for_jerk
    else:
        # RRT/RRT*: constant dt
        d2F_dt2 = np.diff(dF_dt) / dt
        d2tau_dt2 = np.diff(dtau_dt) / dt
    
    if len(d2F_dt2) > 0:
        metrics['force_jerk_rms'] = np.sqrt(np.mean(d2F_dt2**2))
        metrics['torque_jerk_rms'] = np.sqrt(np.mean(d2tau_dt2**2))
        metrics['force_jerk_max'] = np.max(np.abs(d2F_dt2))
        metrics['torque_jerk_max'] = np.max(np.abs(d2tau_dt2))
    else:
        metrics['force_jerk_rms'] = 0.0
        metrics['torque_jerk_rms'] = 0.0
        metrics['force_jerk_max'] = 0.0
        metrics['torque_jerk_max'] = 0.0
    
    # === DIRECTION REVERSALS ===
    metrics['force_reversals'] = np.sum(np.diff(np.sign(dF_dt)) != 0)
    metrics['torque_reversals'] = np.sum(np.diff(np.sign(dtau_dt)) != 0)
    
    # === EXTREME VALUE USAGE ===
    F_max_limit = 14.869296
    tau_max_limit = 0.6351345
    
    metrics['pct_extreme_force'] = (np.sum(np.abs(F) > 0.9 * F_max_limit) / n_steps) * 100
    metrics['pct_extreme_torque'] = (np.sum(np.abs(tau) > 0.9 * tau_max_limit) / n_steps) * 100
    
    # Time at extreme values
    if time_diffs is not None:
        # EST: each command is held for time_diffs[i] seconds
        extreme_force_time = 0
        extreme_torque_time = 0
        for i in range(len(tau) - 1):
            if np.abs(F[i]) > 0.9 * F_max_limit:
                extreme_force_time += time_diffs[i]
            if np.abs(tau[i]) > 0.9 * tau_max_limit:
                extreme_torque_time += time_diffs[i]
        metrics['time_extreme_force'] = extreme_force_time
        metrics['time_extreme_torque'] = extreme_torque_time
    else:
        # RRT/RRT*: simple count * dt
        metrics['time_extreme_force'] = np.sum(np.abs(F) > 0.9 * F_max_limit) * dt
        metrics['time_extreme_torque'] = np.sum(np.abs(tau) > 0.9 * tau_max_limit) * dt
    
    # === TOTAL VARIATION (smoothness) ===
    if time_diffs is not None:
        metrics['force_total_variation'] = np.sum(np.abs(dF_dt) * time_diffs)
        metrics['torque_total_variation'] = np.sum(np.abs(dtau_dt) * time_diffs)
    else:
        metrics['force_total_variation'] = np.sum(np.abs(dF_dt)) * dt
        metrics['torque_total_variation'] = np.sum(np.abs(dtau_dt)) * dt
    
    # === TRAJECTORY INFO ===
    metrics['n_steps'] = n_steps
    metrics['n_commands'] = n_steps  # For both RRT and EST, this is the number of timesteps we're analyzing
    
    if time_diffs is not None:
        metrics['duration'] = np.sum(time_diffs)
        metrics['original_steps'] = original_length
        metrics['compression_ratio'] = compression_ratio
    else:
        metrics['duration'] = n_steps * dt
    
    return metrics


def analyze_trajectories(folder_pattern, algorithm_name='Algorithm'):
    """
    Load and analyze all trajectories in a folder
    """
    files = glob.glob(folder_pattern)
    
    print(f"\n{'='*70}")
    print(f"Analyzing {algorithm_name}")
    print(f"Found {len(files)} files")
    print(f"{'='*70}\n")
    
    if len(files) == 0:
        print(f"WARNING: No files found matching pattern: {folder_pattern}")
        return None, None
    
    all_metrics = []
    skipped = 0
    errors = []
    
    for i, file in enumerate(files):
        try:
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
            
            metrics = compute_aggressiveness_metrics(data)
            
            if metrics is not None:
                all_metrics.append(metrics)
            else:
                skipped += 1
                
        except Exception as e:
            errors.append((i, file, str(e)))
            skipped += 1
    
    print(f"Successfully processed: {len(all_metrics)}/{len(files)}")
    print(f"Skipped: {skipped}")
    
    if errors and len(errors) <= 10:
        print(f"\nErrors encountered in {len(errors)} files:")
        for i, file, error in errors:
            print(f"  File {i}: {error}")
    elif errors:
        print(f"\nErrors encountered in {len(errors)} files (showing first 5):")
        for i, file, error in errors[:5]:
            print(f"  File {i}: {error}")
    
    print()
    
    if len(all_metrics) == 0:
        print("ERROR: No valid metrics computed!")
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Summary statistics on numeric columns only
    summary = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'median': df[numeric_cols].median(),
        'std': df[numeric_cols].std(),
        'q25': df[numeric_cols].quantile(0.25),
        'q75': df[numeric_cols].quantile(0.75),
        'min': df[numeric_cols].min(),
        'max': df[numeric_cols].max(),
        'count': df[numeric_cols].count()
    })
    
    return df, summary


def print_paper_format(summary, title="Results"):
    """
    Print metrics in paper-ready format
    """
    print(f"\n{'='*70}")
    print(f"{title} - Median (IQR)")
    print(f"{'='*70}")
    
    key_metrics = [
        ('torque_jerk_rms', 'Torque Jerk RMS', '[rad/s³]'),
        ('force_jerk_rms', 'Force Jerk RMS', '[N/s²]'),
        ('torque_reversals', 'Torque Reversals', ''),
        ('force_reversals', 'Force Reversals', ''),
        ('pct_extreme_torque', '% Time Extreme τ', '[%]'),
        ('pct_extreme_force', '% Time Extreme F', '[%]'),
        ('time_extreme_torque', 'Time Extreme τ', '[s]'),
        ('torque_total_variation', 'Torque TV', ''),
        ('force_total_variation', 'Force TV', ''),
    ]
    
    for metric_key, metric_label, unit in key_metrics:
        if metric_key in summary.index:
            med = summary.loc[metric_key, 'median']
            q25 = summary.loc[metric_key, 'q25']
            q75 = summary.loc[metric_key, 'q75']
            
            if 'reversals' in metric_key:
                print(f"{metric_label:30s} {unit:10s}: {med:6.0f} ({q25:6.0f}–{q75:6.0f})")
            elif 'pct' in metric_key:
                print(f"{metric_label:30s} {unit:10s}: {med:6.2f} ({q25:6.2f}–{q75:6.2f})")
            else:
                print(f"{metric_label:30s} {unit:10s}: {med:8.3f} ({q25:8.3f}–{q75:8.3f})")
    
    print(f"{'='*70}\n")


def sanity_check_statistics(df, summary, algorithm_name=""):
    """
    Check if statistics make sense
    """
    print("\n" + "="*70)
    print(f"SANITY CHECKS - {algorithm_name}")
    print("="*70)
    
    checks_passed = True
    
    for col in ['force_variance', 'torque_variance']:
        if col in df.columns:
            var = summary.loc[col, 'mean']
            data_col = col.replace('_variance', '')
            min_val = df[f'{data_col}_min'].min()
            max_val = df[f'{data_col}_max'].max()
            
            max_possible_var = ((max_val - min_val) ** 2) / 4
            
            print(f"\n{col}:")
            print(f"  Computed variance: {var:.3f}")
            print(f"  Value range: [{min_val:.3f}, {max_val:.3f}]")
            print(f"  Max possible variance: {max_possible_var:.3f}")
            
            if var > max_possible_var * 1.01:
                print(f"  ❌ FAIL: Variance exceeds theoretical maximum!")
                checks_passed = False
            else:
                print(f"  ✓ OK")
    
    if checks_passed:
        print(f"\n✓ All sanity checks passed!")
    else:
        print(f"\n❌ Some sanity checks FAILED - check your data!")
    
    print("="*70 + "\n")


def investigate_torque_distributions(results_dict):
    """
    Investigate torque value distributions across algorithms
    """
    print("\n" + "="*70)
    print("TORQUE DISTRIBUTION INVESTIGATION")
    print("="*70)
    
    for alg_name, (df, summary) in results_dict.items():
        print(f"\n{alg_name} Statistics:")
        print(f"  Mean abs(τ): {df['torque_mean'].abs().mean():.4f}")
        print(f"  Max abs(τ) across all runs: {df['torque_max'].abs().max():.4f}")
        print(f"  Min abs(τ) across all runs: {df['torque_min'].abs().min():.4f}")
        print(f"  Median trajectory length: {df['n_steps'].median():.0f} steps")
        print(f"  Median duration: {df['duration'].median():.2f} s")
        
        if 'compression_ratio' in df.columns:
            print(f"  Mean compression ratio: {df['compression_ratio'].mean():.1f}x")
            print(f"  (Original {df['original_steps'].median():.0f} steps → {df['n_steps'].median():.0f} commands)")
    
    # Check if EST really uses restricted torques
    if 'EST' in results_dict:
        est_df = results_dict['EST'][0]
        est_max_torque = est_df['torque_max'].abs().max()
        expected_est_max = 0.6351345 / 2
        
        print(f"\nEST Torque Limit Check:")
        print(f"  Expected max (τ_max/2): {expected_est_max:.4f}")
        print(f"  Actual max: {est_max_torque:.4f}")
        
        if est_max_torque > expected_est_max * 1.1:
            print("  ⚠️  WARNING: EST exceeds expected limit!")
        else:
            print("  ✓ EST torque limits as expected")
    
    print("="*70)


def compare_algorithms(results_dict):
    """
    Compare multiple algorithms
    """
    
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON TABLE")
    print("="*80)
    
    key_metrics = [
        ('torque_jerk_rms', 'Torque Jerk RMS'),
        ('torque_reversals', 'Torque Reversals'),
        ('pct_extreme_torque', '% Extreme τ'),
        ('time_extreme_torque', 'Time Extreme τ [s]'),
        ('torque_total_variation', 'Torque TV'),
    ]
    
    # Build comparison table
    comparison = {}
    for alg_name, (df, summary) in results_dict.items():
        comparison[alg_name] = {}
        for metric_key, metric_label in key_metrics:
            if metric_key in summary.index:
                med = summary.loc[metric_key, 'median']
                q25 = summary.loc[metric_key, 'q25']
                q75 = summary.loc[metric_key, 'q75']
                
                if 'reversals' in metric_key:
                    comparison[alg_name][metric_label] = f"{med:.0f} ({q25:.0f}–{q75:.0f})"
                elif 'pct' in metric_key:
                    comparison[alg_name][metric_label] = f"{med:.1f} ({q25:.1f}–{q75:.1f})"
                else:
                    comparison[alg_name][metric_label] = f"{med:.3f} ({q25:.3f}–{q75:.3f})"
    
    comp_df = pd.DataFrame(comparison).T
    print(comp_df.to_string())
    print()
    
    # Statistical comparisons
    if len(results_dict) >= 2:
        print("="*80)
        print("STATISTICAL COMPARISONS (Mann-Whitney U test)")
        print("="*80)
        
        alg_names = list(results_dict.keys())
        
        for i in range(len(alg_names)):
            for j in range(i + 1, len(alg_names)):
                alg1 = alg_names[i]
                alg2 = alg_names[j]
                
                df1 = results_dict[alg1][0]
                df2 = results_dict[alg2][0]
                
                print(f"\n{alg1} vs {alg2}:")
                print("-" * 80)
                
                for metric_key, metric_label in key_metrics:
                    if metric_key in df1.columns and metric_key in df2.columns:
                        vals1 = df1[metric_key].dropna()
                        vals2 = df2[metric_key].dropna()
                        
                        if len(vals1) > 0 and len(vals2) > 0:
                            stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
                            
                            med1 = vals1.median()
                            med2 = vals2.median()
                            
                            if med1 != 0:
                                diff_pct = ((med2 - med1) / med1) * 100
                            else:
                                diff_pct = 0
                            
                            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                            
                            print(f"{metric_label:25s}: {alg1}={med1:7.3f}, {alg2}={med2:7.3f}, "
                                  f"diff={diff_pct:+6.1f}%, p={p:.4f} {sig}")
    
    return comp_df


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    results = {}
    
    # ========================================================================
    # ANALYZE RRT
    # ========================================================================
    print("\n" + "#"*80)
    print("# RRT ANALYSIS")
    print("#"*80)
    
    rrt_folder = '/home/dimitria/PhD_codes/quadcopter_load/RRT_paths_2D_obstacle_5/*.yaml'
    rrt_df, rrt_summary = analyze_trajectories(rrt_folder, 'RRT Scenario 1')
    
    if rrt_summary is not None:
        print("\nFULL SUMMARY STATISTICS:")
        print(rrt_summary.to_string())
        
        print_paper_format(rrt_summary, "RRT Scenario 1")
        sanity_check_statistics(rrt_df, rrt_summary, "RRT")
        
        # Save results
        rrt_summary.to_csv('rrt_scenario_1_summary.csv')
        rrt_df.to_csv('rrt_scenario_1_all_metrics.csv', index=False)
        
        print(f"RRT results saved to:")
        print(f"  - rrt_scenario_1_summary.csv")
        print(f"  - rrt_scenario_1_all_metrics.csv\n")
        
        results['RRT'] = (rrt_df, rrt_summary)
    
    # ========================================================================
    # ANALYZE RRT*
    # ========================================================================
    print("\n" + "#"*80)
    print("# RRT* ANALYSIS")
    print("#"*80)
    
    rrt_star_folder = '/home/dimitria/PhD_codes/quadcopter_load/RRT_star_paths_2D_obstacle_5/*.yaml'
    rrt_star_df, rrt_star_summary = analyze_trajectories(rrt_star_folder, 'RRT* Scenario 1')
    
    if rrt_star_summary is not None:
        print("\nFULL SUMMARY STATISTICS:")
        print(rrt_star_summary.to_string())
        
        print_paper_format(rrt_star_summary, "RRT* Scenario 1")
        sanity_check_statistics(rrt_star_df, rrt_star_summary, "RRT*")
        
        # Save results
        rrt_star_summary.to_csv('rrt_star_scenario_1_summary.csv')
        rrt_star_df.to_csv('rrt_star_scenario_1_all_metrics.csv', index=False)
        
        print(f"RRT* results saved to:")
        print(f"  - rrt_star_scenario_1_summary.csv")
        print(f"  - rrt_star_scenario_1_all_metrics.csv\n")
        
        results['RRT*'] = (rrt_star_df, rrt_star_summary)
    
    # ========================================================================
    # ANALYZE EST
    # ========================================================================
    print("\n" + "#"*80)
    print("# EST ANALYSIS")
    print("#"*80)
    
    est_folder = '/home/dimitria/PhD_codes/quadcopter_load/info_obstacle_5_2/*.yaml'
    est_df, est_summary = analyze_trajectories(est_folder, 'EST Scenario 1')
    
    if est_summary is not None:
        print("\nFULL SUMMARY STATISTICS:")
        print(est_summary.to_string())
        
        print_paper_format(est_summary, "EST Scenario 1")
        sanity_check_statistics(est_df, est_summary, "EST")
        
        # Save results
        est_summary.to_csv('est_scenario_1_summary.csv')
        est_df.to_csv('est_scenario_1_all_metrics.csv', index=False)
        
        print(f"EST results saved to:")
        print(f"  - est_scenario_1_summary.csv")
        print(f"  - est_scenario_1_all_metrics.csv\n")
        
        results['EST'] = (est_df, est_summary)
    
    # ========================================================================
    # MULTI-ALGORITHM COMPARISON
    # ========================================================================
    if len(results) >= 2:
        print("\n" + "#"*80)
        print("# MULTI-ALGORITHM COMPARISON")
        print("#"*80)
        
        # Investigate torque distributions
        investigate_torque_distributions(results)
        
        # Compare algorithms
        comparison_table = compare_algorithms(results)
        comparison_table.to_csv('algorithm_comparison_scenario_1.csv')
        print(f"\nComparison table saved to: algorithm_comparison_scenario_1.csv")
        
        # Create a summary for the paper
        print("\n" + "="*80)
        print("SUMMARY FOR PAPER - TABLE III")
        print("="*80)
        print("\nValues shown as Median (IQR)\n")
        print(comparison_table.to_string())
        
        # Export to LaTeX
        print("\n" + "="*80)
        print("LaTeX TABLE")
        print("="*80)
        print(comparison_table.to_latex())
        
        # Additional insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        if 'EST' in results and 'RRT' in results:
            est_df = results['EST'][0]
            rrt_df = results['RRT'][0]
            
            est_cmds = est_df['n_commands'].median()
            rrt_cmds = rrt_df['n_commands'].median()
            
            est_dur = est_df['duration'].median()
            rrt_dur = rrt_df['duration'].median()
            
            print(f"\nCommand Frequency:")
            print(f"  RRT: ~{rrt_cmds:.0f} commands in {rrt_dur:.2f}s = {rrt_cmds/rrt_dur:.1f} commands/s")
            print(f"  EST: ~{est_cmds:.0f} commands in {est_dur:.2f}s = {est_cmds/est_dur:.1f} commands/s")
            
            if 'original_steps' in est_df.columns:
                print(f"\nEST Downsampling:")
                print(f"  Original: {est_df['original_steps'].median():.0f} steps")
                print(f"  After removing zeros: {est_cmds:.0f} actual commands")
                print(f"  Compression: {est_df['compression_ratio'].median():.1f}x")
            
            est_jerk = est_df['torque_jerk_rms'].median()
            rrt_jerk = rrt_df['torque_jerk_rms'].median()
            print(f"\nSmoothness:")
            print(f"  EST jerk: {est_jerk:.1f} rad/s³")
            print(f"  RRT jerk: {rrt_jerk:.1f} rad/s³")
            print(f"  EST is {(1-est_jerk/rrt_jerk)*100:.1f}% smoother")
    
    else:
        print("\nNot enough algorithms analyzed for comparison")
    
    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print("#"*80)
    print(f"\nProcessed {len(results)} algorithms successfully")
    print("Check the CSV files for detailed results")