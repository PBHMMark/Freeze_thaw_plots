import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats, signal

def process_data(input_file, output_file):
    # Read in the data
    df = pd.read_csv(input_file)

    # Convert time column to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Add year and month columns
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    # Count number of days where tasmin < 0 and tasmax > 0
    df_count = df[(df['tasmin'] < 0) & (df['tasmax'] > 0)].groupby(['year', 'month']).size().reset_index(name='count')

    # Save to a csv file
    df_count.to_csv(output_file, index=False)


def heatmap_plotter(counts_df):
    # Colour palette
    colors = ['#FFFFFF', '#9CBDC0', '#0A3341']
    cmap_name = 'my_list'
    n_bins = 1000
    pal = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Create a new column with the date
    counts_df["date"] = pd.to_datetime(counts_df[['year', 'month']].assign(day=1))

    # Pivot the DataFrame to get the counts per year per month
    pivot_counts = counts_df.pivot_table(values="count", index="year", columns="month")

    # Create a heatmap plot
    plt.figure(figsize=(10, 10))
    sns.heatmap(pivot_counts, cmap=pal, annot=False, cbar_kws={"label": "Count"})
    plt.title("Freeze-Thaw Events per Year per Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.savefig("C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/heatmap_col.png")


def plot_by_month(counts_df):
    # Create a new column with the date
    counts_df["date"] = pd.to_datetime(counts_df[['year', 'month']].assign(day=1))

    # Loop through each month and create a separate plot
    for month in range(1, 13):
        # Filter the data for the current month
        month_counts = counts_df[counts_df["month"] == month]

        # Create a scatterplot with a line of best fit
        sns.lmplot(x="year", y="count", data=month_counts, height=5, aspect=1.5)

        # Set the plot title and axis labels
        plt.title("Freeze-Thaw Events in Month {}".format(month))
        plt.xlabel("Year")
        plt.ylabel("Count")

        # Save the plot to file
        plt.savefig("C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/month{}_plot.png".format(month))


def plot_by_month_single(counts_df):
    # Create a new column with the date
    counts_df["date"] = pd.to_datetime(counts_df[['year', 'month']].assign(day=1))

    # Create a 4x3 grid of plots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))

    # Loop through each month and create a separate plot
    for i, month in enumerate(range(1, 13)):
        # Filter the data for the current month
        month_counts = counts_df[counts_df["month"] == month]

        # Create a scatterplot with a line of best fit
        sns.regplot(x="year", y="count", data=month_counts, ax=axes[i//3, i%3], color='#9CBDC0', line_kws={'color':'#0A3341'}, scatter_kws={'s': 20, 'alpha': 1, 'facecolor': '#9CBDC0', 'edgecolor': '#9CBDC0'})

        # Set the plot title and axis labels
        axes[i//3, i%3].set_title("Freeze-Thaw Events in Month {}".format(month))
        axes[i//3, i%3].set_xlabel("Year")
        axes[i//3, i%3].set_ylabel("Count")

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot to file
    plt.savefig("C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/month_plots_single.png")

def plot_freeze_thaw_trend(counts_df, window_size=10):
    # Aggregate counts by year
    df = counts_df.groupby('year')['count'].sum().reset_index()

    # Calculate moving average
    df['moving_avg'] = df['count'].rolling(window_size).mean()

    # Calculate linear trendline
    x = df['year'].values
    y = df['count'].values
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    residuals = y - (slope * x + intercept)
    ss_resid = np.sum(residuals ** 2)
    ss_total = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_resid / ss_total)
    trendline = intercept + slope * x

    # Plot data and trendlines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['year'], df['count'], c='#9CBDC0', s=40, label='Freeze Thaw Days')
    ax.plot(df['year'], df['moving_avg'], color='#0A3341', linewidth=2, label=f'{window_size}-year Moving Average')
    ax.plot(df['year'], trendline, color='#AD832D', linewidth=2, label='Linear Trendline', zorder=10)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count of Freeze Thaw Days')
    ax.set_title('Trend of Freeze Thaw Days')

    # Add equation and R-squared value to plot
    eqn_text = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}'
    ax.text(1.1, 0.5, eqn_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#9CBDC0', linewidth=1))

    # Move legend outside of plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',  edgecolor='#9CBDC0')

    plt.tight_layout()
    plt.savefig("C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/yeartrendfig.png")

if __name__ == '__main__':
    # Set input and output file paths
    input_file = r'C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/ahccd (2).csv'
    output_file = r'C:/PhD/Courses/ENGO697/Data/HistoricData/Freeze_Thaw/Rawdata/counts.csv'

    # Process the data and save to file
    process_data(input_file, output_file)

    # Read in the processed data
    counts_df = pd.read_csv(output_file)

    # Create plots
    heatmap_plotter(counts_df)
    #plot_by_month(counts_df)
    #plot_by_month_single(counts_df)
    #plot_freeze_thaw_trend(counts_df,5)

