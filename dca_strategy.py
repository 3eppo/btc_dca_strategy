import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def prepare_data(file_path):
    """
    Load Bitcoin price data.
    """
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

def simulate_hourly_dca(data, amount=100):
    """
    Simulate hourly DCA purchases.
    """
    purchases = data.resample('H').first().dropna()
    purchases['btc_bought'] = amount / purchases['Open']
    purchases['cost'] = amount
    avg_price = purchases['cost'].sum() / purchases['btc_bought'].sum()
    return purchases, avg_price

def simulate_daily_dca(data, hour, amount=100):
    """
    Simulate daily DCA purchases at a specific hour.
    """
    purchases = data[data.index.hour == hour].copy()
    purchases['btc_bought'] = amount / purchases['Open']
    purchases['cost'] = amount
    avg_price = purchases['cost'].sum() / purchases['btc_bought'].sum()
    return purchases, avg_price

def simulate_weekly_dca(data, weekday, hour, amount=100):
    """
    Simulate weekly DCA purchases on a specific weekday and hour.
    """
    purchases = data[(data.index.dayofweek == weekday) & (data.index.hour == hour)].copy()
    purchases['btc_bought'] = amount / purchases['Open']
    purchases['cost'] = amount
    avg_price = purchases['cost'].sum() / purchases['btc_bought'].sum()
    return purchases, avg_price

def generate_heatmaps(data, amount=100):
    """
    Generate heatmaps for DCA strategies.
    """
    # Heatmap for daily and hourly strategies
    daily_hourly_results = []
    for day in range(7):
        for hour in range(24):
            _, avg_price = simulate_weekly_dca(data, day, hour, amount)
            daily_hourly_results.append({'Day': day, 'Hour': hour, 'Avg_Price': avg_price})

    daily_hourly_df = pd.DataFrame(daily_hourly_results)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_hourly_df['Day'] = daily_hourly_df['Day'].apply(lambda x: day_order[x])
    heatmap_daily_hourly = daily_hourly_df.pivot(index='Day', columns='Hour', values='Avg_Price')

    heatmap_daily_hourly = heatmap_daily_hourly.reindex(day_order[::-1])

    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_daily_hourly, annot=False, cmap='coolwarm', cbar_kws={'label': 'Average Purchase Price'})
    plt.title('Heatmap of Weekly DCA Strategies (Mon to Sun)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.show(block=False)

    # Heatmap for hourly strategies only
    hourly_results = []
    for hour in range(24):
        _, avg_price = simulate_daily_dca(data, hour, amount)
        hourly_results.append({'Hour': hour, 'Avg_Price': avg_price})

    hourly_df = pd.DataFrame(hourly_results)
    hourly_df = hourly_df.sort_values(by='Hour', ascending=False)
    plt.figure(figsize=(6, 8))
    sns.heatmap(hourly_df.set_index('Hour'), annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Average Purchase Price'})
    plt.title('Heatmap of Hourly DCA Strategies')
    plt.xlabel('')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.show(block=False)

def visualize_bar_chart(results, title="Overall Top Strategies"):
    """
    Visualize the top strategies with a bar chart.
    """
    top_30 = results.head(30).copy()
    best_price = top_30['Avg_Price'].min()

    # Calculate relative deviation
    top_30['Relative_Deviation'] = ((top_30['Avg_Price'] - best_price) / best_price) * 100

    plt.figure(figsize=(10, 8))
    bars = plt.barh(
        [f"{row['Strategy']} ({row['Relative_Deviation']:.2f}% Dev.)" for _, row in top_30.iterrows()],
        top_30['Relative_Deviation'],
        color='skyblue'
    )

    plt.xlabel('Deviation from Best Price (%)')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show(block=False)

def visualize_filtered_bar_chart(results):
    """
    Visualize the top strategies without weekly strategies.
    """
    filtered_results = results[~results['Strategy'].str.startswith('Weekly')]
    visualize_bar_chart(filtered_results, title="Top Strategies (Excluding Weekly)")

def main():
    file_path = "btc_data_hourly.csv"

    # Load data
    data = prepare_data(file_path)

    # Show min and max dates
    min_date = data.index.min()
    max_date = data.index.max()
    print(f"Data available from {min_date.date()} to {max_date.date()}.")

    # Get user input for date range
    while True:
        try:
            start_date = input(f"Enter the start date (YYYY-MM-DD) between {min_date.date()} and {max_date.date()}: ")
            end_date = input(f"Enter the end date (YYYY-MM-DD) between {min_date.date()} and {max_date.date()}: ")
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

            if min_date <= start_date <= max_date and min_date <= end_date <= max_date and start_date <= end_date:
                break
            else:
                print("Please ensure the dates are within the available range and the start date is before the end date.")
        except ValueError:
            print("Invalid date format. Please enter dates in YYYY-MM-DD format.")

    # Filter data
    filtered_data = data.loc[start_date:end_date]

    # Compare strategies
    results = []

    # Weekly strategies
    for day in range(7):
        for hour in range(24):
            _, avg_price = simulate_weekly_dca(filtered_data, day, hour, amount=100)
            results.append({
                'Strategy': f'Weekly {"Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()[day]} {hour}:00',
                'Avg_Price': avg_price
            })

    # Daily strategies
    for hour in range(24):
        _, avg_price = simulate_daily_dca(filtered_data, hour, amount=100)
        results.append({
            'Strategy': f'Daily {hour}:00',
            'Avg_Price': avg_price
        })

    # Hourly strategy
    _, avg_price = simulate_hourly_dca(filtered_data, amount=100)
    results.append({'Strategy': 'Hourly', 'Avg_Price': avg_price})

    results_df = pd.DataFrame(results).sort_values(by='Avg_Price')
    results_df['Relative_Deviation'] = ((results_df['Avg_Price'] - results_df['Avg_Price'].min()) / results_df['Avg_Price'].min()) * 100
    results_df.to_csv(f'dca_strategy_results_{start_date.date()}_to_{end_date.date()}.csv', index=False)
    print(f"Results saved to 'dca_strategy_results_{start_date.date()}_to_{end_date.date()}.csv'")

    # Visualize results
    plt.figure()
    visualize_bar_chart(results_df)

    plt.figure()
    visualize_filtered_bar_chart(results_df)

    plt.figure()
    generate_heatmaps(filtered_data, amount=100)

    plt.show()

if __name__ == "__main__":
    main()
