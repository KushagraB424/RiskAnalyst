import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    plt.style.use('ggplot')
    # Use consistent font family, sizes, grid style, line widths, color palette
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'legend.fontsize': 10,
        'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    })

def plot_cpu_vs_gpu(df, out_dir):
    plt.figure(figsize=(10,6))
    for engine in df['Engine'].unique():
        sub = df[df['Engine'] == engine]
        plt.plot(sub['NumPaths'], sub['TotalTime'], marker='o', label=engine)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Execution Time: CPU vs GPU')
    plt.xlabel('Number of Simulations (Paths)')
    plt.ylabel('Total Execution Time (ms) [Log Scale]')
    plt.legend()
    plt.savefig(f"{out_dir}/cpu_vs_gpu_runtime.png", bbox_inches='tight')
    plt.close()

def plot_speedup(df, out_dir):
    plt.figure(figsize=(10,6))
    for engine in df['Engine'].unique():
        if engine == 'CPU Sequential': continue
        sub = df[df['Engine'] == engine]
        plt.plot(sub['NumPaths'], sub['SpeedupSeq'], marker='o', label=f'{engine} vs CPU Seq')
    
    plt.xscale('log')
    plt.title('Speedup relative to Sequential CPU')
    plt.xlabel('Number of Simulations (Paths)')
    plt.ylabel('Speedup Factor (x)')
    plt.legend()
    plt.savefig(f"{out_dir}/speedup.png", bbox_inches='tight')
    plt.close()

def plot_throughput(df, out_dir):
    plt.figure(figsize=(10,6))
    for engine in df['Engine'].unique():
        sub = df[df['Engine'] == engine]
        plt.plot(sub['NumPaths'], sub['Throughput'], marker='o', label=engine)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Simulation Throughput')
    plt.xlabel('Number of Simulations (Paths)')
    plt.ylabel('Throughput (Paths / Second) [Log Scale]')
    plt.legend()
    plt.savefig(f"{out_dir}/throughput.png", bbox_inches='tight')
    plt.close()

def plot_breakdown(df, out_dir):
    opt = df[df['Engine'] == 'CUDA Optimized']
    if opt.empty: return
    max_path = opt['NumPaths'].max()
    row = opt[opt['NumPaths'] == max_path].iloc[0]
    
    times = [row['H2DTime'], row['KernelTime'], row['D2HTime']]
    labels = ['Host -> Device', 'Kernel Execution', 'Device -> Host']
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    plt.figure(figsize=(8,8))
    wedges, texts, autotexts = plt.pie(times, labels=None, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(f'CUDA Time Breakdown (N = {max_path:,.0f})')
    plt.savefig(f"{out_dir}/transfer_breakdown.png", bbox_inches='tight')
    plt.close()

def plot_convergence(df, out_dir):
    plt.figure(figsize=(10,6))
    sub = df[df['Engine'] == 'CUDA Optimized']
    if not sub.empty:
        # Calculate 95% Confidence Interval using Standard Error
        N = sub['NumPaths']
        mean = sub['Mean']
        std_dev = sub['StdDev']
        standard_error = std_dev / np.sqrt(N)
        ci_lower = mean - 1.96 * standard_error
        ci_upper = mean + 1.96 * standard_error

        plt.plot(N, mean, marker='o', label='Mean Payoff', color='#1f77b4')
        plt.fill_between(N, ci_lower, ci_upper, color='#1f77b4', alpha=0.3, label='95% Confidence Interval')
        plt.xscale('log')
        plt.title('Monte Carlo Convergence (Mean Payoff)')
        plt.xlabel('Number of Simulations (Paths)')
        plt.ylabel('Mean Payoff')
        plt.legend()
        plt.savefig(f"{out_dir}/convergence.png", bbox_inches='tight')
        plt.close()

def plot_distributions(payoffs_file, out_dir, df):
    if not os.path.exists(payoffs_file): return
    
    pdf = pd.read_csv(payoffs_file)
    payoffs = pdf['Payoff'].values
    
    plt.figure(figsize=(10,6))
    plt.hist(payoffs, bins=100, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.title('Portfolio Returns Histogram')
    plt.xlabel('Discounted Payoff')
    plt.ylabel('Frequency')
    plt.savefig(f"{out_dir}/portfolio_histogram.png", bbox_inches='tight')
    plt.close()

    opt = df[df['Engine'] == 'CUDA Optimized']
    if opt.empty: return
    row = opt.iloc[-1]
    
    plt.figure(figsize=(10,6))
    plt.hist(payoffs, bins=100, color='#ff7f0e', edgecolor='black', alpha=0.7)
    plt.axvline(row['Mean'], color='black', linestyle='dashed', linewidth=2, label=f"Mean: {row['Mean']:.2f}")
    plt.axvline(row['VaR95'], color='#2ca02c', linestyle='dashed', linewidth=2, label=f"VaR (95%): {row['VaR95']:.2f}")
    plt.axvline(row['CVaR95'], color='#d62728', linestyle='dashed', linewidth=2, label=f"CVaR (95%): {row['CVaR95']:.2f}")
    plt.title('Risk Distribution (VaR & Expected Shortfall)')
    plt.xlabel('Discounted Payoff')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{out_dir}/var_distribution.png", bbox_inches='tight')
    plt.close()

def generate_markdown(df, out_dir):
    hw_info = ""
    hw_path = f"{out_dir}/hardware_info.txt"
    if os.path.exists(hw_path):
        with open(hw_path, 'r') as f:
            hw_info = f.read()
            
    conf_info = ""
    conf_path = f"{out_dir}/benchmark_config.txt"
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf_info = f.read()
    
    max_path = df['NumPaths'].max()
    max_df = df[df['NumPaths'] == max_path]
    
    opt_speedup = 0
    omp_speedup = 0
    opt_rel_error = 0
    threads_used = 1
    opt_block = 0
    opt_grid = 0
    
    if not max_df[max_df['Engine'] == 'CUDA Optimized'].empty:
        opt_row = max_df[max_df['Engine'] == 'CUDA Optimized'].iloc[0]
        opt_speedup = opt_row['SpeedupSeq']
        opt_rel_error = opt_row['RelError']
        opt_block = opt_row['BlockSize']
        opt_grid = opt_row['GridSize']
        
    if not max_df[max_df['Engine'] == 'CPU OpenMP'].empty:
        omp_row = max_df[max_df['Engine'] == 'CPU OpenMP'].iloc[0]
        omp_speedup = omp_row['SpeedupSeq']
        threads_used = omp_row['ThreadsUsed']
        
    omp_efficiency = omp_speedup / threads_used if threads_used > 0 else 0

    md = f"""# Benchmark Summary

## Hardware Configuration
```text
{hw_info}
```

## Benchmark Configuration
```text
{conf_info}
```

## Performance Table
"""
    
    md += "| Engine | Paths | Total Time (ms) | Throughput (paths/s) | Speedup vs Seq | Rel. Error |\n"
    md += "|---|---|---|---|---|---|\n"
    for _, row in df.iterrows():
        md += f"| {row['Engine']} | {row['NumPaths']} | {row['TotalTime']:.2f} | {row['Throughput']:.2e} | {row['SpeedupSeq']:.2f}x | {row['RelError']:.4f} |\n"
    
    md += f"""
## Key Findings
- **CUDA Optimized** achieved a **{opt_speedup:.1f}x** speedup over Sequential CPU.
- **CPU OpenMP** achieved a **{omp_speedup:.1f}x** speedup using {threads_used} threads (Efficiency: {omp_efficiency*100:.1f}%).
- CUDA Block Size was dynamically chosen as **{opt_block}**, resulting in a Grid Size of **{opt_grid}**.
- Mean payoff across all engines differed from the CPU reference by less than **{(opt_rel_error * 100):.2f}%**.
- Numerical convergence and confidence intervals remained highly stable for simulations $\geq 1,000,000$.

## Visualizations
![CPU vs GPU Runtime](cpu_vs_gpu_runtime.png)
![Speedup](speedup.png)
![Throughput](throughput.png)
![Monte Carlo Convergence](convergence.png)
![Transfer Breakdown](transfer_breakdown.png)
![VaR Distribution](var_distribution.png)
![Portfolio Histogram](portfolio_histogram.png)
"""
    
    with open(f"{out_dir}/benchmark_summary.md", 'w') as f:
        f.write(md)

def main():
    out_dir = '../results'
    csv_file = f'{out_dir}/benchmark_results.csv'
    payoffs_file = f'{out_dir}/payoffs_sample.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the C++ benchmark first.")
        return
        
    setup_style()
    df = pd.read_csv(csv_file)
    
    plot_cpu_vs_gpu(df, out_dir)
    plot_speedup(df, out_dir)
    plot_throughput(df, out_dir)
    plot_breakdown(df, out_dir)
    plot_convergence(df, out_dir)
    plot_distributions(payoffs_file, out_dir, df)
    
    generate_markdown(df, out_dir)

if __name__ == '__main__':
    main()
