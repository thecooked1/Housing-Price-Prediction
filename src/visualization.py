import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Global styling
sns.set_theme(style="whitegrid")

def _save_and_clean(filename: str):
    """
    Helper function to save the plot and close the figure to prevent memory issues.
    """
    output_dir = "../reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_distribution(df: pd.DataFrame, column: str):
    """
    Create a distribution plot (Histogram + KDE) for a numerical column.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the numerical column to visualize.

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    KeyError
        If the specified column is not found in the DataFrame.
    """
    try:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")

        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, color='teal')
        plt.title(f'Distribution Analysis: {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        _save_and_clean(f"distribution_{column}.png")
    except Exception as e:
        print(f"Error plotting distribution for {column}: {e}")
        raise

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Create a heatmap to visualize correlations between numerical features.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing numerical features.

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    Exception
        If an error occurs during numeric selection or plotting.
    """
    try:
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=['number'])
        correlation = numeric_df.corr()

        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')

        _save_and_clean("correlation_heatmap.png")
    except Exception as e:
        print(f"Error plotting correlation heatmap: {e}")
        raise

def plot_scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Create a scatter plot with a regression trend line.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    x_col : str
        The independent variable (x-axis).
    y_col : str
        The dependent variable (y-axis).

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    KeyError
        If specified columns are not found in the DataFrame.
    """
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError("Specified columns not found in DataFrame.")

        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=x_col, y=y_col,
                    scatter_kws={'alpha': 0.3, 'color': 'gray'},
                    line_kws={'color': 'red'})
        plt.title(f'Relationship Analysis: {x_col} vs {y_col}')

        _save_and_clean(f"scatter_{x_col}_vs_{y_col}.png")
    except Exception as e:
        print(f"Error plotting scatter plot: {e}")
        raise

def plot_categorical_counts(df: pd.DataFrame, column: str):
    """
    Create a bar chart showing the frequency of categories.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The categorical column to analyze.

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    KeyError
        If the specified column is not found in the DataFrame.
    """
    try:
        # Check if the column exists in categorical columns
        if column not in df.columns:
            # Note: We check columns containing 'ocean_proximity' because of One-Hot Encoding
            potential_cols = [c for c in df.columns if column in c]
            if not potential_cols:
                raise KeyError(f"No columns matching '{column}' found.")

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, hue=column, palette='viridis', legend=False)
        plt.title(f'Frequency of {column}')
        plt.xticks(rotation=45)

        _save_and_clean(f"counts_{column}.png")
    except Exception as e:
        print(f"Error plotting count plot for {column}: {e}")
        raise

def plot_outlier_boxplot(df: pd.DataFrame, columns: list):
    """
    Create box plots for visual outlier detection across multiple columns.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list;
        List of numerical columns to check for outliers.

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    KeyError
        If any of the specified columns are missing.
    """
    try:
        plt.figure(figsize=(12, 6))
        # Melt dataframe to plot multiple numerical columns together
        melted_df = df[columns].melt()
        sns.boxplot(x='variable', y='value', hue='variable', data=melted_df, legend=False)
        plt.title('Outlier Analysis of Key Variables')
        plt.xticks(rotation=45)

        _save_and_clean("outlier_boxplots.png")
    except Exception as e:
        print(f"Error plotting boxplots: {e}")
        raise

def plot_violin_by_category(df: pd.DataFrame, cat_col: str, num_col: str):
    """
    Create a violin plot to show numerical density distribution across categories.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    cat_col : str
        The categorical variable (x-axis).
    num_col : str
        The numerical variable (y-axis).

    Returns:
    ----------
    None
        Saves the generated plot to the reports/figures directory.

    Raises:
    ----------
    KeyError
        If specified columns are missing.
    """
    try:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x=cat_col, y=num_col, hue=cat_col, palette='muted', legend=False)
        plt.title(f'{num_col} Density by {cat_col}')

        _save_and_clean(f"violin_{cat_col}_{num_col}.png")
    except Exception as e:
        print(f"Error plotting violin plot: {e}")
        raise