import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weekly_comparison(city_dfs, poi_locations, poi_type):
    """
    Plot weekly flow patterns comparing the same POI type across different cities
    """
    colors = {
        'CityA': '#FF9999',
        'CityB': '#66B2FF',
        'CityC': '#99FF99',
        'CityD': '#FFCC99'
    }
    
    # Calculate maximum number of weeks across all cities
    max_weeks = 0
    for city, df in city_dfs.items():
        locations = poi_locations[poi_type][city]
        for x_target, y_target in locations:
            subset = df[(df["x"] == x_target) & (df["y"] == y_target)]
            n_days = subset["d"].max() + 1
            n_weeks = int(np.ceil(n_days / 7))
            max_weeks = max(max_weeks, n_weeks)

    # Setup subplots (2 columns)
    n_cols = 2
    n_rows = int(np.ceil(max_weeks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows), sharey=True)
    axes = axes.flatten()

    # Plot each week
    for week in range(max_weeks):
        ax = axes[week]
        
        # Plot data for each city
        for city, df in city_dfs.items():
            locations = poi_locations[poi_type][city]
            
            # Handle multiple locations (e.g., Nagoya stations)
            for loc_idx, (x_target, y_target) in enumerate(locations):
                subset = df[(df["x"] == x_target) & (df["y"] == y_target)].copy()
                subset["hour_index"] = subset["d"] * 24 + subset["t"] / 2

                start_day = week * 7
                end_day = min((week + 1) * 7, subset["d"].max() + 1)
                
                if start_day >= end_day:
                    continue
                    
                mask = (subset["d"] >= start_day) & (subset["d"] < end_day)
                week_data = subset[mask].copy()
                week_data["hour_index"] = (week_data["d"] - start_day) * 24 + week_data["t"] / 2
                week_series = week_data.groupby("hour_index").size().reset_index(name="count")

                # Create full hour range
                full_hours = pd.DataFrame({"hour_index": np.arange(0, 7*24, 0.5)})
                week_series = full_hours.merge(week_series, on="hour_index", how="left").fillna(0)

                # Different style for multiple locations of same city
                alpha = 0.8 if len(locations) == 1 else [0.8, 0.4][loc_idx]
                label = f'{city}' if len(locations) == 1 or loc_idx == 0 else f'{city}_Location2'
                
                ax.plot(week_series["hour_index"], 
                       week_series["count"],
                       color=colors[city],
                       alpha=alpha,
                       label=label if week == 0 else "")  # Only show legend in first subplot

        ax.set_title(f"Week {week+1}")
        ax.set_xlim(0, 168)

        # Set ticks
        xticks = []
        xticklabels = []
        weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        for d in range(7):
            base = d * 24
            for h in [0, 6, 12, 18]:
                xticks.append(base + h)
                if h == 0:
                    xticklabels.append(f"0({weekdays[d]})")
                else:
                    xticklabels.append(str(h))
        xticks.append(7*24)
        xticklabels.append("24")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)
        if week == 0:  # Only show legend in first subplot
            ax.legend()

    # Remove extra subplots
    for j in range(max_weeks, len(axes)):
        fig.delaxes(axes[j])

    # Labels
    fig.suptitle(f"Weekly Flow Comparison - {poi_type}", y=1.02, fontsize=14)
    for ax in axes:
        if ax.get_subplotspec().get_geometry()[2] % n_cols == 0:  # First column
            ax.set_ylabel("Flow count")
        if ax.get_subplotspec().get_geometry()[2] >= len(axes) - n_cols:  # Last row
            ax.set_xlabel("Hour within week (0-24)")

    plt.tight_layout()
    plt.show()

def main():
    # Read city data
    city_dfs = {
        'CityA': pd.read_csv('task1_dataset_kotae.csv'),
        'CityB': pd.read_csv('hiroshima_challengedata(B).csv'),
        'CityC': pd.read_csv('sapporo_challengedata(C).csv'),
        'CityD': pd.read_csv('kumamoto_challengedata(D).csv')
    }

    # POI locations
    poi_locations = {
        'university': {
            'CityA': [(136,77)],
            'CityB': [(80,94)],
            'CityC': [(24,148)],
            'CityD': [(100,104)]
        },
        'station': {
            'CityA': [(135,77), (134,83)],  # Two stations in Nagoya
            'CityB': [(79,93)],
            'CityC': [(24,151)],
            'CityD': [(101,103)]
        },
        'shopping': {
            'CityA': [(134,82)],
            'CityB': [(79,93)],
            'CityC': [(26,153)],
            'CityD': [(101,102)]
        }
    }

    # Plot each POI type comparison
    for poi_type in ['university', 'station', 'shopping']:
        print(f"Plotting {poi_type} comparison...")
        plot_weekly_comparison(city_dfs, poi_locations, poi_type)

if __name__ == "__main__":
    main()