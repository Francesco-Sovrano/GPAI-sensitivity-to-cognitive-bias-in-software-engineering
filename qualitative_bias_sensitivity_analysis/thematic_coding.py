import re
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

results_dir = "./analysis_results"
os.makedirs(results_dir, exist_ok=True)

# Define theme keywords
themes_keywords = {
    'performance': ['availability', 'efficient', 'latency', 'optimize', 'load', 'scalability', 'speed', 'optimization', 'performance', 'improve', 'quick', 'efficiency', 'fast', 'processing', 'asynchronous', 'bandwidth', 'stability', 'downtime', 'productivity', 'throughput', 'compression', 'reliability', 'multiplexing', 'scalable', 'slow', 'benchmark', 'grow', 'better', 'exponential'],
    'security': ['protection', 'vulnerabilities', 'intrusion', 'risk', 'privacy', 'encryption', 'access', 'robustness', 'vulnerability', 'auth', 'cve', 'audit', 'attack', 'security', 'compliance', 'secure', 'authorization', 'threat', 'gdpr', 'logging', 'monitoring'],
    'team': ['collaboration', 'expertise', 'agility', 'skill', 'staff', 'training', 'meetings', 'management', 'manageability', 'refactor', 'team', 'onboarding', 'staffing', 'maintainability', 'mentorship', 'organization', 'knowledge', 'capability', 'communicate'],
    'customer': ['communicat', 'community', 'functionality', 'company', 'stakeholder', 'feature', 'customer', 'satisfaction', 'user experience', 'usability', 'feedback', 'client', 'reputation', 'industry', 'market', 'user', 'maintenance', 'business', 'retention', 'commerce'],
    'cost': [
        'overhead', 'cost', 'sacrificing',
        'investment',  # already present
        'resource'     # “resource allocation,” “resource limits”
    ],
    'timeline': ['delivery', 'lag', 'time', 'late', 'immediately', 'schedule', 'scheduling', 'wait', 'longer', 'roadmap', 'delay', 'deadline', 'milestone', 'now', 'urgent', 'quick', 'crucial', 'cut-off', 'prompt', 'immediate', 'promptly', 'fast', 'quick', 'slow', 'term'],
    'trendiness': ['state-of-the-art', 'prototype', 'community', 'cutting-edge', 'latest', 'adoption', 'modern', 'trend', 'emerging', 'common', 'new library', 'beta', 'innovative', 'up-to-date'],
    'maintainability': ['deprecated', 'technical debt', 'manageable', 'maintained', 'maintain', 'manageability', 'refactor', 'maintainable', 'maintenance', 'refactoring', 'maintainability', 'maintaining', 'logging', 'deployment']
,
    'reliability': ['failure', 'vulnerabilities', 'instability', 'consistency', 'robustness', 'redundancy', 'fail-safe', 'uptime', 'resilience', 'rollback', 'robust', 'maintainability', 'reliability', 'fault-tolerant'],
    'scalability': ['load balancing', 'scale', 'shard', 'elasticity', 'limits', 'scalability', 'optimized', 'performance', 'complexity', 'optimizations', 'manageability', 'throttling', 'stability', 'maintainability', 'scalable', 'containerization', 'scaling'],
    'quality': [
      'test',         # unit, integration, smoke, etc.
      'testing',      # verb/noun form
      'debugging',
      'verification',
      'validation',
      'qa',
    ],
    'usability': ['customization', 'intuitive', 'ux', 'design', 'documentation', 'ui', 'ergonomic', 'experience', 'interface', 'frustrating'],
    'compliance': ['policies', 'security', 'audit', 'compliance', 'audits', 'guidelines', 'oversight', 'accountability', 'policy', 'documentation', 'regulation', 'practices', 'standard', 'hipaa', 'gdpr'],
    'effort': ['learning', 'challenging', 'straightforward', 'ramp-up', 'effort', 'implementing', 'making', 'optimized', 'training', 'hand-roll', 'simplicity', 'improving', 'easier', 'improves', 'complexity', 'optimizations', 'simpler', 'staffing', 'improvements', 'manual'],
}

def assign_themes(text):
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [theme for theme, kws in themes_keywords.items() if any(k in t for k in kws)]

# File pattern to match all relevant CSVs\ nfile_pattern = 'generated_data/2_llm_outputs_model=*.data_model_list=[[]*[]].csv'
file_pattern = '../generated_data/2_llm_outputs_model=*.data_model_list=[[]*[]].csv'
files = glob.glob(file_pattern)

for file_path in files+['bias_sample_analysis.csv']:
    # Extract model name from filename
    base = os.path.basename(file_path)
    match = re.search(r"outputs_model=(.*?)\.data_model_list=", base)
    model_name = match.group(1) if match else base.replace('.csv', '')

    # Load and filter data
    df = pd.read_csv(file_path)
    if 'sensitive_to_bias' in df.columns:
        df = df[df['sensitive_to_bias']]

    # Annotate themes
    for suffix in ['without_bias', 'with_bias']:
        col = f"decision_explanation_{suffix}"
        if col in df.columns:
            df[f"themes_{suffix}"] = df[col].apply(assign_themes)

    # Save thematic coding
    out = df[[
        'decision_explanation_without_bias',
        'decision_explanation_with_bias',
        'themes_without_bias',
        'themes_with_bias'
    ]]
    csv_out = f"thematic_coding_{model_name}.csv"
    out.to_csv(os.path.join(results_dir, csv_out), index=False)
    print(f"Saved {csv_out}")

    # Prepare counts for bar chart
    counts = {
        theme: (
            df['themes_without_bias'].explode().value_counts().get(theme, 0),
            df['themes_with_bias'].explode().value_counts().get(theme, 0)
        ) for theme in sorted(themes_keywords)
    }
    labels, without_vals, with_vals = zip(*[(t, v[0], v[1]) for t, v in counts.items()])
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, without_vals, width, label='Without Bias')
    ax.bar(x + width/2, with_vals, width, label='With Bias')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_xlabel('')
    # ax.set_title('Theme Frequencies')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for i, (w, b) in enumerate(zip(without_vals, with_vals)):
        ax.text(i - width/2, w + 0.1, str(w), ha='center', va='bottom')
        ax.text(i + width/2, b + 0.1, str(b), ha='center', va='bottom')
    plt.tight_layout()
    bar_out = f"theme_frequency_{model_name}.pdf"
    fig.savefig(os.path.join(results_dir, bar_out), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved bar chart: {bar_out}")

heatmaps = []
bias_sample_heatmap = None

for file_path in files + ['bias_sample_analysis.csv']:
    base = os.path.basename(file_path)
    match = re.search(r"outputs_model=(.*?)\.data_model_list=", base)
    model_name = match.group(1) if match else base.replace('.csv', '')

    df = pd.read_csv(file_path)
    if 'sensitive_to_bias' in df.columns:
        df = df[df['sensitive_to_bias']]

    for suffix in ['without_bias', 'with_bias']:
        col = f"decision_explanation_{suffix}"
        if col in df.columns:
            df[f"themes_{suffix}"] = df[col].apply(assign_themes)

    if 'bias_name' in df.columns:
        melted = df.melt(
            id_vars='bias_name',
            value_vars=['themes_without_bias', 'themes_with_bias'],
            var_name='code_type', value_name='themes'
        ).explode('themes').dropna()

        melted['type'] = melted['code_type'].map({
            'themes_without_bias': 'without',
            'themes_with_bias': 'with'
        })

        grouped = melted.groupby(['themes', 'bias_name', 'type']).size().unstack(fill_value=0)
        diff_df = (grouped.get('with', pd.Series()) - grouped.get('without', pd.Series())).unstack(fill_value=0)
        diff_df = diff_df.fillna(0)

        if 'bias_sample_analysis.csv' in file_path:
            bias_sample_heatmap = ('bias_sample_analysis', diff_df)
        else:
            heatmaps.append((model_name, diff_df))

# === Plot combined heatmaps (excluding bias_sample_analysis) ===
n = len(heatmaps)
cols = 2
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), sharex=True, sharey=False)

if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1 or cols == 1:
    axes = axes.reshape((rows, cols))

for i, (ax, (model_name, diff_df)) in enumerate(zip(axes.flat, heatmaps)):
    is_last_col = (i % cols == cols - 1)
    is_first_col = (i % cols == 0)
    sns.heatmap(
        diff_df,
        annot=True,
        fmt='d',
        cmap='icefire',
        cbar=True,
        cbar_kws={'label': 'Δ Count'} if is_last_col else None,
        ax=ax
    )
    ax.set_title(model_name)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if is_first_col: # Always show x and y tick labels explicitly
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    else:
        ax.set_yticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# fig.suptitle("Δ Theme Counts Across Bias Types (All Models)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(
    wspace=0.05,  # reduce horizontal space
    hspace=0.15   # reduce vertical space
)
combined_out = os.path.join(results_dir, "combined_heatmaps_shared_axes.pdf")
fig.savefig(combined_out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved combined heatmap figure: {combined_out}")

# === Plot separate heatmap for bias_sample_analysis.csv ===
if bias_sample_heatmap:
    model_name, diff_df = bias_sample_heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        diff_df,
        annot=True,
        fmt='d',
        cmap='icefire',
        cbar=True,
        cbar_kws={'label': 'Δ Count'},
        ax=ax2
    )
    ax2.set_title("Bias Sample Analysis")
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    plt.tight_layout()
    sep_out = os.path.join(results_dir, "heatmap_bias_sample_analysis.pdf")
    fig2.savefig(sep_out, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved separate heatmap: {sep_out}")
