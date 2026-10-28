"""
eda_figures.py
──────────────
Generates all EDA figures for the breast cancer classification report.
Outputs saved to ./figs/eda/

Loads data directly from ./splits/ — just run the cell as-is.
Requires the splits directory to contain:
    X_train.npy, y_train.npy, feature_index.npy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

# ── Output directory ──────────────────────────────────────────────────────────
FIG_DIR = './figs/eda'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BENIGN_COL     = '#4C72B0'
MALIGNANT_COL  = '#DD8452'
ENGINEERED_COL = '#DD8452'
RAW_COL        = '#4C72B0'

sns.set_theme(style='whitegrid', font_scale=1.0)

# ── Load data ─────────────────────────────────────────────────────────────────
SPLIT_DIR     = './splits/'
X_train       = np.load(f'{SPLIT_DIR}X_train.npy')
y_train       = np.load(f'{SPLIT_DIR}y_train.npy')
feature_index = np.load(f'{SPLIT_DIR}feature_index.npy', allow_pickle=True).item()

# ── Build a labelled DataFrame from the full scaled training array ────────────
# feature_index maps every feature name → its column position in X_train
feature_names = list(feature_index.keys())
X_train_df    = pd.DataFrame(X_train, columns=feature_names)

# ── Selected and engineered feature names ─────────────────────────────────────
SELECTED = [
    'scale__mean_radius',
    'scale__tumor_size_log',
    'scale__morphological_irregularity',
    'scale__scaled_border_texture',
]

ENGINEERED = [
    'scale__radial_concavity', # radius * concavity
    'scale__radial_border_texture', # radius * smoothness
]

RAW_COMPONENTS = [
    'scale__mean_radius',
    'scale__tumor_size_log',
    'scale__smoothness',
    'scale__concavity',
]

# ── Pre-compute univariate AUC for all features ───────────────────────────────
aucs = {}
for col in X_train_df.columns:
    auc = roc_auc_score(y_train, X_train_df[col])
    aucs[col] = max(auc, 1 - auc)   # flip negatively-correlated features

auc_series = pd.Series(aucs).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Class Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig1_class_distribution():
    y_series = pd.Series(y_train)
    counts    = y_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ['Benign (0)', 'Malignant (1)'],
        counts.values,
        color=[BENIGN_COL, MALIGNANT_COL],
        edgecolor='white', width=0.5
    )
    for bar, count in zip(bars, counts.values):
        pct = count / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )
    ax.set_title('Figure 1: Class Distribution (Training Set)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 1 saved')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Univariate AUC Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig2_univariate_auc():
    display_names = [n.replace('scale__', '') for n in auc_series.index]
    colours = [ENGINEERED_COL if n in SELECTED else RAW_COL
               for n in auc_series.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(display_names, auc_series.values, color=colours, edgecolor='white')

    ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random (0.5)')
    ax.axvline(0.8, color='red',  linestyle='--', linewidth=0.8, label='Strong signal (0.8)')

    ax.set_xlabel('Univariate AUC')
    ax.set_title(
        'Figure 2: Univariate AUC by Feature\n(orange = selected features)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0.45, 1.02)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_univariate_auc.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 2 saved')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Correlation Heatmap of Raw Components
# ─────────────────────────────────────────────────────────────────────────────
def fig3_correlation_heatmap():
    label_map = {col: col.replace('scale__', '') for col in RAW_COMPONENTS}
    corr = (X_train_df[RAW_COMPONENTS]
            .corr()
            .rename(columns=label_map, index=label_map))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True, fmt='.2f',
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        ax=ax
    )
    ax.set_title('Figure 3: Correlation Matrix — Raw Feature Components',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 3 saved')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Violin Plots: Raw Components vs Engineered Features
# ─────────────────────────────────────────────────────────────────────────────
def fig4_raw_vs_engineered_violin():
    compare = {
        'smoothness\n(raw)':                        'scale__smoothness',
        'concavity\n(raw)':                         'scale__concavity',
        'radial_concavity\n(engineered)':      'scale__radial_concavity',
        'radial_border_texture\n(engineered)': 'scale__radial_border_texture',
    }

    plot_df = X_train_df.copy()
    plot_df['diagnosis'] = pd.Series(y_train).astype(str).values

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (ax, (label, col)) in enumerate(zip(axes, compare.items())):
        sns.violinplot(
            data=plot_df, x='diagnosis', y=col,
            hue='diagnosis',
            palette={'0': BENIGN_COL, '1': MALIGNANT_COL},
            inner='quartile', ax=ax, linewidth=0.8,
            legend=False
        )
        auc_val = aucs.get(col)
        title_str = f'{label}\nAUC = {auc_val:.3f}' if auc_val else label
        ax.set_title(title_str, fontsize=9, fontweight='bold')
        ax.set_xlabel('Diagnosis', fontsize=8)
        ax.set_ylabel('Scaled Value' if i == 0 else '', fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        if 'engineered' in label:
            ax.set_facecolor('#f9f9f9')

    fig.suptitle(
        'Figure 4: Class Separation — Raw Components vs. Engineered Features',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_raw_vs_engineered_violin.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 4 saved')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Incremental AUC over mean_radius Baseline
# ─────────────────────────────────────────────────────────────────────────────
def fig5_incremental_auc():
    baseline_col   = 'scale__mean_radius'
    candidate_cols = {
        'tumor_size_log (raw)':                    'scale__tumor_size_log',
        'smoothness (raw)':                        'scale__smoothness',
        'concavity (raw)':                         'scale__concavity',
        'radial_concavity (engineered)':      'scale__radial_concavity',
        'radial_border_texture (engineered)': 'scale__radial_border_texture',
    }

    def quick_auc(cols):
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_df[cols].values, y_train,
            cv=5, scoring='roc_auc'
        )
        return scores.mean()

    baseline_auc = quick_auc([baseline_col])
    print(f'  Baseline AUC (mean_radius only): {baseline_auc:.4f}')

    results = []
    for label, col in candidate_cols.items():
        combined_auc = quick_auc([baseline_col, col])
        gain = combined_auc - baseline_auc
        results.append({
            'Feature Added':              label,
            'Baseline AUC':               round(baseline_auc, 4),
            'Combined AUC':               round(combined_auc, 4),
            'AUC Gain':                   round(gain, 4),
        })
        print(f'  + {label:<45} → AUC {combined_auc:.4f}  (gain: {gain:+.4f})')

    inc_df = pd.DataFrame(results).sort_values('AUC Gain', ascending=False)

    colours = [
        ENGINEERED_COL if 'engineered' in r else RAW_COL
        for r in inc_df['Feature Added']
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(
        inc_df['Feature Added'], inc_df['AUC Gain'],
        color=colours, edgecolor='white'
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('AUC Gain over mean_radius Baseline')
    ax.set_title(
        'Figure 5: Incremental AUC Gain When Added to mean_radius\n'
        '(orange = retained engineered features)',
        fontsize=12, fontweight='bold'
    )
    for bar, val in zip(bars, inc_df['AUC Gain']):
        offset = 0.0005 if val >= 0 else -0.0005
        ha     = 'left'  if val >= 0 else 'right'
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.4f}', va='center', ha=ha, fontsize=9
        )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig5_incremental_auc.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 5 saved')

    return inc_df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Partial Correlation (controlling for mean_radius)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_partial_correlation():
    control = 'scale__mean_radius'
    candidates = {
        'tumor_size_log (raw)':                    'scale__tumor_size_log',
        'smoothness (raw)':                        'scale__smoothness',
        'concavity (raw)':                         'scale__concavity',
        'radial_concavity (engineered)':      'scale__radial_concavity',
        'radial_border_texture (engineered)': 'scale__radial_border_texture',
    }

    def partial_corr(feature_col, control_col, target):
        """Correlation between feature and target after residualising out control."""
        X_ctrl = X_train_df[[control_col]].values

        resid_feat   = (X_train_df[feature_col].values
                        - LinearRegression().fit(X_ctrl, X_train_df[feature_col]).predict(X_ctrl))
        resid_target = (np.array(target)
                        - LinearRegression().fit(X_ctrl, target).predict(X_ctrl))

        return pd.Series(resid_feat).corr(pd.Series(resid_target))

    y_arr = np.array(y_train, dtype=float)
    partial_results = {
        label: round(partial_corr(col, control, y_arr), 4)
        for label, col in candidates.items()
    }

    partial_series = (pd.Series(partial_results)
                      .reindex(sorted(partial_results, key=lambda x: abs(partial_results[x]),
                                      reverse=True)))

    print('\nPartial correlations with diagnosis (controlling for mean_radius):')
    print(partial_series.to_string())

    colours = [
        ENGINEERED_COL if 'engineered' in label else RAW_COL
        for label in partial_series.index
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(partial_series.index, partial_series.values,
            color=colours, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Partial Correlation with Diagnosis')
    ax.set_title(
        'Figure 6: Partial Correlation with Diagnosis\n'
        '(controlling for mean_radius — orange = retained engineered features)',
        fontsize=12, fontweight='bold'
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig6_partial_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 6 saved')

    return partial_series


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Stratified Violin Plots (within mean_radius bins)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_stratified_violin():
    plot_df = X_train_df.copy()
    plot_df['diagnosis'] = pd.Series(y_train).astype(str).values
    plot_df['radius_bin'] = pd.qcut(
        plot_df['scale__mean_radius'],
        q=3,
        labels=['Low\nmean_radius', 'Mid\nmean_radius', 'High\nmean_radius']
    )

    engineered = {
        'radial_concavity':      'scale__radial_concavity',
        'radial_border_texture': 'scale__radial_border_texture',
    }
    bin_labels = ['Low\nmean_radius', 'Mid\nmean_radius', 'High\nmean_radius']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')

    for row, (feat_label, feat_col) in enumerate(engineered.items()):
        for col_idx, bin_label in enumerate(bin_labels):
            ax       = axes[row, col_idx]
            bin_data = plot_df[plot_df['radius_bin'] == bin_label]

            sns.violinplot(
                data=bin_data, x='diagnosis', y=feat_col,
                hue='diagnosis',
                palette={'0': BENIGN_COL, '1': MALIGNANT_COL},
                inner='quartile', ax=ax, linewidth=0.8,
                legend=False
            )

            if bin_data['diagnosis'].nunique() == 2:
                bin_auc = roc_auc_score(
                    bin_data['diagnosis'].astype(int),
                    bin_data[feat_col]
                )
                bin_auc = max(bin_auc, 1 - bin_auc)
                ax.set_title(f'{bin_label}\nAUC = {bin_auc:.3f}', fontsize=9)
            else:
                ax.set_title(bin_label, fontsize=9)

            ax.set_xlabel('Diagnosis', fontsize=8)
            ax.set_xticklabels(['Benign', 'Malignant'], fontsize=8)
            ax.set_ylabel(feat_label if col_idx == 0 else '', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        'Figure 7: Engineered Feature Class Separation Within mean_radius Bins\n'
        '(persistent AUC across bins confirms independent signal)',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig7_stratified_violin.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✓ Figure 7 saved')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — AUC Summary Table (print + save as CSV)
# ─────────────────────────────────────────────────────────────────────────────
def table1_auc_summary():
    summary_features = {
        'mean_radius (raw)':                          'scale__mean_radius',
        'tumor_size_log (raw)':                       'scale__tumor_size_log',
        'smoothness (raw)':                           'scale__smoothness',
        'concavity (raw)':                            'scale__concavity',
        'radial_concavity (engineered)':      'scale__radial_concavity',
        'radial_border_texture (engineered)': 'scale__radial_border_texture',
    }

    rows = []
    for label, col in summary_features.items():
        rows.append({
            'Feature':         label,
            'Type':            'Engineered' if 'engineered' in label else 'Raw',
            'Univariate AUC':  round(aucs.get(col, float('nan')), 4),
        })

    table_df = pd.DataFrame(rows).sort_values('Univariate AUC', ascending=False)
    print('\nTable 1: AUC Summary — Raw vs Engineered Features')
    print(table_df.to_string(index=False))
    table_df.to_csv(f'{FIG_DIR}/table1_auc_summary.csv', index=False)
    print('✓ Table 1 saved')

    return table_df


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__' or True:
    print('=' * 60)
    print('Generating EDA figures → ./figs/eda/')
    print('=' * 60)

    fig1_class_distribution()
    fig2_univariate_auc()
    fig3_correlation_heatmap()
    fig4_raw_vs_engineered_violin()
    fig5_incremental_auc()
    fig6_partial_correlation()
    fig7_stratified_violin()
    table1_auc_summary()

    print('\n' + '=' * 60)
    print('All figures saved to ./figs/eda/')
    print('=' * 60)
