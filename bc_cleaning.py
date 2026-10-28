import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DuplicateHandler:
    def process(self, df):
        df['diagnosis'] = df['diagnosis'].replace({'B': 'Benign'}).str.title() # Standardise labels
        df = df.groupby('patient_id', as_index=False, sort=False).agg(self._resolve_conflicts) # Resolve id duplicates
        return df

    def _resolve_conflicts(self, series): # Collapse duplicates to one row per patient
        s = series.dropna().unique()  # get unique non-null value
        return s[0] if len(s) == 1 else np.nan  # conflicts → NaN


class OutlierHandler:
    def process(self, df):
        for col in ( # Z-score capping for approximately normal features
            'mean_radius', 'mean_texture', 'mean_perimeter', 'age',
            'smoothness', 'compactness', 'symmetry', 'concavity'):
            df[col] = self._z_score_cap(df[col])

        df['mean_area'] = self._iqr_cap(df['mean_area'])
        df['tumor_size_mm'] = self._z_score_cap(np.log1p(df['tumor_size_mm']))
        df.rename(columns={'tumor_size_mm': 'tumor_size_log'}, inplace=True)

        numeric_cols = df.select_dtypes(include='number').columns.drop('patient_id')
        df[numeric_cols] = df[numeric_cols].clip(lower=0) # domain based clip
        return df

    def _iqr_cap(self, series):
        q1, q3 = series.quantile([0.25, 0.75]) 
        iqr = q3 - q1
        return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    def _z_score_cap(self, series, threshold=3):
        mean, std = series.mean(), series.std()
        return series.clip(mean - threshold * std, mean + threshold * std)


class ImputationHandler:
    def process(self, df):
        self._impute_categoricals(df)
        self._impute_numerics(df)
        return df
    
    def _fill_mode(self, x): # fill missing values with mode of the series
        m = x.mode() 
        return x.fillna(m[0]) if not m.empty else x # if empty, return original series
    
    def _impute_categoricals(self, df): 
        age_groups = pd.cut(df['age'], bins=range(0, 121, 10)) # create bins for age

        df['menopause_status'] = (
            df.groupby(age_groups, observed=True)['menopause_status']
            .transform(self._fill_mode).fillna(df['menopause_status'].mode().iloc[0])
        ) # impute menopause_status via age bins — hormone stage correlates strongly with age

        cat_cols = df.select_dtypes(include='object').columns.drop(
            ['genetic_mutation', 'diagnosis', 'menopause_status']) # exclude target and 
        
        df[cat_cols] = (
            df.groupby('menopause_status', observed=True)[cat_cols]
            .transform(self._fill_mode).fillna(df[cat_cols].mode().iloc[0])
        ) # hormone stage influences many other features, so use menopause_status as grouping 

        df['genetic_mutation'] = df['genetic_mutation'].fillna('Unknown') # High missingness, fill with 'Unknown'
        return df
    
    def _impute_numerics(self, df):
        numeric_cols = df.select_dtypes(include='number').columns.drop(['patient_id']) # exlcude identifier
        df[numeric_cols] = ( 
            df.groupby('menopause_status', observed=True)[numeric_cols]
            .transform(lambda x: x.fillna(x.mean())) # impute with mean within menopause groups
            .fillna(df[numeric_cols].mean()) # global mean imputation as fallback
        )
        return df
    


def print_categorical_distribution(df):
    for c in df.select_dtypes(include='object').columns:
        counts = df[c].value_counts().items()
        pcts = {k: f"{v/3500:.0%}" for k, v in counts}
        print(f"{c:<25} : {pcts}")

def show_numerical_histogram(df, filename):
    num_cols = df.select_dtypes(include='number').columns.drop('patient_id')
    df[num_cols].hist(bins=30, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.savefig(f'./figs/{filename}.png')
    plt.show()

def missing_values_analysis(df, filename):
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    missing_pct.plot(kind='bar')

    plt.title("Percentage of Missing Values per Column")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Columns")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(f'./figs/{filename}.png')
    plt.show()
    missing = df.isnull()

    # Keep only columns with missing values
    cols_with_missing = missing.columns[missing.sum() > 0]

    # If nothing is missing, skip plot
    if len(cols_with_missing) == 0:
        print("No missing values to plot.")
        return

    missing_subset = missing[cols_with_missing]
    missing_corr = missing_subset.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(missing_corr, dtype=bool))
    masked_corr = missing_corr.mask(mask)

    plt.figure(figsize=(8, 6))

    im = plt.imshow(masked_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)

    # Axis labels
    plt.xticks(range(len(cols_with_missing)), cols_with_missing, rotation=45, ha='right')
    plt.yticks(range(len(cols_with_missing)), cols_with_missing)

    # Annotate values
    for i in range(len(cols_with_missing)):
        for j in range(len(cols_with_missing)):
            if not mask[i, j] and not np.isnan(masked_corr.iloc[i, j]):
                plt.text(j, i, f"{masked_corr.iloc[i, j]:.2f}",
                         ha='center', va='center', fontsize=8)

    plt.title("Missingness Correlation Between Columns")
    plt.tight_layout()
    plt.savefig(f'./figs/{filename}.png')
    plt.show()
    
def show_categorical_distribution(df, filename, title):
    # Identify categorical columns (object or category types)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) == 0:
        print("No categorical columns found.")
        return

    # Calculate grid dimensions (3 columns wide)
    cols_per_row = 3
    n_rows = int(np.ceil(len(cat_cols) / cols_per_row))

    # Now n_rows is exactly an int (e.g., 3), not a float (3.0)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 5 * n_rows))
    axes = axes.flatten() # Flatten to iterate easily

    for i, col in enumerate(cat_cols):
        # Create count plot
        sns.countplot(y=col, data=df, ax=axes[i], palette='viridis', hue=col, legend=False)
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Count')
        axes[i].set_ylabel('') # Removing y-label for cleaner look
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add global title and optimize spacing
    plt.suptitle(title, fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f'./figs/{filename}.png', bbox_inches='tight')
    plt.show()

def plot_individual_missing_pies(df):
    """
    Generates an individual pie chart for each feature containing missing values.
    Optimized into a 3-column widescreen grid layout to maximize horizontal space.
    """
    # 1. Identify columns that have missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    
    if not missing_cols:
        print("No missing values found. Skipping visualization.")
        return
    
    # 2. Configure grid layout to use 3 columns for widescreen layout
    n_cols = len(missing_cols)
    grid_cols = 3  # Changed to 3 to spread out horizontally
    grid_rows = math.ceil(n_cols / grid_cols)
    
    # Expanded width to 16 (widescreen aspect ratio) and adjusted row height
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 4.5 * grid_rows))
    
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    total_rows = len(df)
    
    # 3. Plot a dedicated pie chart for each column
    for i, col in enumerate(missing_cols):
        missing = missing_counts[col]
        present = total_rows - missing
        
        sizes = [missing, present]
        labels = ['Missing', 'Present']
        colors = ['#d9534f', '#5cb85c']  # Red for missing, Green for present
        
        # Clean up column technical names for presentation
        clean_title = col.replace('_', ' ').title()
        
        axes[i].pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
            textprops={'fontsize': 10},
            pctdistance=0.65  # Keeps percentages neatly inside the slices
        )
        
        # Add titles with clean vertical padding
        axes[i].set_title(f"{clean_title}\n({missing:,} / {total_rows:,} missing)", 
                          fontsize=11, fontweight='bold', pad=10)
        
    # Hide any empty subplot boxes if the total number doesn't perfectly divide by 3
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Adjust padding for the new 3-column structure
    plt.tight_layout(pad=2.5, h_pad=4.0, w_pad=3.0)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('./breast_cancer_dataset.csv')
    # missing_values_analysis(df, 'missing_values_before_cleaning')

    # show_numerical_histogram(df, 'breast_cancer_histogram')

    # print(f"{df.info()}\n{df.head().T}\n{df.describe()}\n")  # inspect data set
    # print_categorical_distribution(df)

    # numeric_cols = df.select_dtypes(include='number').columns.drop('patient_id')
    # for col in numeric_cols:
    #     stat, p = stats.shapiro(df[col].dropna())
    #     print(f"{col}: W={stat:.4f}, p={p:.4f}") 
    # plot_individual_missing_pies(df)

    id_dupes = df[df['patient_id'].duplicated(keep=False)].sort_values('patient_id')
    print(f"Number of IDs with duplicates: {id_dupes['patient_id'].nunique()}\n")
    print(f"Diagnosis value counts before standardisation:\n{df['diagnosis'].value_counts()}\n")

    df = DuplicateHandler().process(df)

    id_dupes = df[df['patient_id'].duplicated(keep=False)].sort_values('patient_id')
    print(f"Number of IDs with duplicates: {id_dupes['patient_id'].nunique()}\n")
    print(f"Diagnosis value counts after standardisation and duplicates resolved:\n{df['diagnosis'].value_counts()}\n")

    # print(f"{df.info()}\n{df.head().T}\n{df.describe()}\n")  # inspect data set
    # print_categorical_info(df)



    df = OutlierHandler().process(df)
    df = ImputationHandler().process(df)
    print(f"Missing values per column after cleaning:\n{df.isnull().sum()}\n")


    # print_categorical_distribution(df)
    # print(f"{df.info()}\n{df.head().T}\n{df.describe()}\n")  # inspect data set

    print(f"Number of records after cleaning: {df.shape[0]}")

    df.to_csv('./cleaned_breast_cancer_dataset.csv', index=False)
    print("Cleaned data saved to ./cleaned_breast_cancer_dataset.csv")

    # show_numerical_histogram(df, 'cleaned_breast_cancer_histogram_mean')
    # show_categorical_distribution(df, 'cleaned_breast_cancer_categorical_distribution', 'Categorical Distribution')
