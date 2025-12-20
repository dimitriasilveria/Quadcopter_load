import numpy as np
import glob
import pandas as pd
from scipy.stats import mannwhitneyu

def compute_load_metrics(load_angles, load_angular_vel, dt=0.01):
    theta = np.array(load_angles)
    theta_dot = np.array(load_angular_vel)
    
    if len(theta) < 3:
        return None
    
    metrics = {}
    
    # === LOAD ANGLE STATISTICS ===
    metrics['max_load_angle'] = np.max(np.abs(theta))
    metrics['mean_abs_load_angle'] = np.mean(np.abs(theta))
    metrics['rms_load_angle'] = np.sqrt(np.mean(theta**2))
    
    # === LOAD ANGULAR VELOCITY STATISTICS ===
    metrics['max_load_angular_vel'] = np.max(np.abs(theta_dot))
    metrics['rms_load_angular_vel'] = np.sqrt(np.mean(theta_dot**2))
    metrics['mean_abs_load_angular_vel'] = np.mean(np.abs(theta_dot))
    
    # === LOAD ANGULAR ACCELERATION ===
    theta_ddot = np.diff(theta_dot) / dt
    
    metrics['load_angular_accel_rms'] = np.sqrt(np.mean(theta_ddot**2))
    metrics['load_angular_accel_max'] = np.max(np.abs(theta_ddot))
    metrics['mean_abs_load_angular_accel'] = np.mean(np.abs(theta_ddot))
    
    # === OSCILLATION METRICS ===
    metrics['load_angle_reversals'] = np.sum(np.diff(np.sign(theta_dot)) != 0)
    
    # === EXTREME SWING EVENTS ===
    large_swing_threshold_rad = np.deg2rad(20)
    metrics['pct_time_large_swing'] = (np.sum(np.abs(theta) > large_swing_threshold_rad) / len(theta)) * 100
    metrics['time_large_swing'] = np.sum(np.abs(theta) > large_swing_threshold_rad) * dt
    
    violent_accel_threshold = 5.0
    metrics['num_violent_accel_events'] = np.sum(np.abs(theta_ddot) > violent_accel_threshold)
    
    # === TRAJECTORY INFO ===
    metrics['n_steps'] = len(theta)
    metrics['duration'] = len(theta) * dt
    
    return metrics


def analyze_load_swing(method='RRT', obstacle=1):
    folder = f"/home/dimitria/PhD_codes/quadcopter_load/states/{method}_obstacle_{obstacle}_states/*.npz"
    states_files = glob.glob(folder)
    
    print(f"\n{'='*70}")
    print(f"Analyzing Load Swing - {method} Obstacle {obstacle}")
    print(f"Found {len(states_files)} files")
    print(f"{'='*70}\n")
    
    if len(states_files) == 0:
        print(f"WARNING: No files found!")
        return None, None
    
    all_metrics = []
    skipped = 0
    
    for file in states_files:
        try:
            data = np.load(file)
            
            if 'states' not in data:
                skipped += 1
                continue
            
            states = data['states']
            
            if len(states) == 0:
                skipped += 1
                continue
            
            if len(states) < 3:
                skipped += 1
                continue
            
            # Extract load angles and angular velocities
            load_angles = []
            load_angular_vel = []
            
            for state in states:
                # Handle both RRT format (N, 8) and EST format (N, 8, 1)
                if state.ndim > 1:
                    # EST format - squeeze to remove extra dimension
                    state_flat = np.squeeze(state)
                else:
                    # RRT format - already flat
                    state_flat = state
                
                if len(state_flat) >= 6:
                    load_angles.append(float(state_flat[4]))
                    load_angular_vel.append(float(state_flat[5]))
            
            if len(load_angles) < 3:
                skipped += 1
                continue
            
            metrics = compute_load_metrics(load_angles, load_angular_vel)
            
            if metrics is not None:
                all_metrics.append(metrics)
            else:
                skipped += 1
                
        except:
            skipped += 1
    
    print(f"Successfully processed: {len(all_metrics)}/{len(states_files)}")
    print(f"Skipped: {skipped}\n")
    
    if len(all_metrics) == 0:
        print("ERROR: No valid data found!")
        return None, None
    
    df = pd.DataFrame(all_metrics)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
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


def print_load_swing_summary(summary, title="Load Swing Analysis"):
    print(f"\n{'='*70}")
    print(f"{title} - Median (IQR)")
    print(f"{'='*70}")
    
    key_metrics = [
        ('load_angular_accel_rms', 'Load Accel RMS', '[rad/s²]'),
        ('load_angular_accel_max', 'Max Load Accel', '[rad/s²]'),
        ('max_load_angle', 'Max Load Angle', '[rad]'),
        ('rms_load_angle', 'RMS Load Angle', '[rad]'),
        ('load_angle_reversals', 'Angle Reversals', ''),
        ('pct_time_large_swing', '% Time Large Swing', '[%]'),
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
            elif 'angle' in metric_key.lower() and 'accel' not in metric_key:
                med_deg = np.rad2deg(med)
                q25_deg = np.rad2deg(q25)
                q75_deg = np.rad2deg(q75)
                print(f"{metric_label:30s} {unit:10s}: {med_deg:.2f}° ({q25_deg:.2f}°–{q75_deg:.2f}°)")
            else:
                print(f"{metric_label:30s} {unit:10s}: {med:8.4f} ({q25:8.4f}–{q75:8.4f})")
    
    print(f"{'='*70}\n")


def compare_load_swing(results_dict):
    print("\n" + "="*80)
    print("LOAD SWING COMPARISON")
    print("="*80)
    
    key_metrics = [
        ('load_angular_accel_rms', 'Load Accel RMS'),
        ('max_load_angle', 'Max Angle [deg]'),
        ('rms_load_angle', 'RMS Angle [deg]'),
        ('load_angle_reversals', 'Reversals'),
    ]
    
    comparison = {}
    for alg_name, (df, summary) in results_dict.items():
        comparison[alg_name] = {}
        for metric_key, metric_label in key_metrics:
            if metric_key in summary.index:
                med = summary.loc[metric_key, 'median']
                q25 = summary.loc[metric_key, 'q25']
                q75 = summary.loc[metric_key, 'q75']
                
                if 'angle' in metric_key and 'accel' not in metric_key:
                    med = np.rad2deg(med)
                    q25 = np.rad2deg(q25)
                    q75 = np.rad2deg(q75)
                
                if 'reversals' in metric_key:
                    comparison[alg_name][metric_label] = f"{med:.0f} ({q25:.0f}–{q75:.0f})"
                elif 'deg' in metric_label:
                    comparison[alg_name][metric_label] = f"{med:.2f} ({q25:.2f}–{q75:.2f})"
                else:
                    comparison[alg_name][metric_label] = f"{med:.3f} ({q25:.3f}–{q75:.3f})"
    
    comp_df = pd.DataFrame(comparison).T
    print(comp_df.to_string())
    print()
    
    if len(results_dict) >= 2:
        print("="*80)
        print("STATISTICAL COMPARISONS")
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
                            
                            if 'angle' in metric_key and 'accel' not in metric_key:
                                med1 = np.rad2deg(med1)
                                med2 = np.rad2deg(med2)
                            
                            diff_pct = ((med2 - med1) / med1 * 100) if med1 != 0 else 0
                            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                            
                            print(f"{metric_label:25s}: {alg1}={med1:7.3f}, {alg2}={med2:7.3f}, diff={diff_pct:+6.1f}%, p={p:.4f} {sig}")
    
    return comp_df


if __name__ == "__main__":
    obstacle = 1
    results = {}
    
    for method_name in ['RRT', 'RRT_star', 'EST']:
        df, summary = analyze_load_swing(method=method_name, obstacle=obstacle)
        
        if summary is not None:
            print_load_swing_summary(summary, f"{method_name}")
            summary.to_csv(f'{method_name.lower()}_load_swing.csv')
            results[method_name] = (df, summary)
    
    if len(results) >= 2:
        comparison_table = compare_load_swing(results)
        comparison_table.to_csv(f'load_swing_comparison.csv')
        print("\nLaTeX TABLE:")
        print(comparison_table.to_latex())
    
    print("\nDONE")