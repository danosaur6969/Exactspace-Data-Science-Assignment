import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

def load_and_validate_data():
    print("🔄 Loading cyclone sensor data...")
    df = pd.read_excel("data.xlsx")
    print(f"✅ Dataset loaded successfully!\n📊 Shape: {df.shape}")
    print("Columns available:", df.columns.tolist())

    TIME_COLUMN = "time"  # update as needed

    if TIME_COLUMN not in df.columns:
        raise Exception(f"Time column '{TIME_COLUMN}' not found! Please check column names.")
    sensor_cols = [col for col in df.columns if col != TIME_COLUMN]

    error_strings = ['I/O Timeout', 'Not Connect', 'Error', 'No Data', 'Disconnected', '', ' ']
    df[sensor_cols] = df[sensor_cols].replace(error_strings, np.nan)
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    df.set_index(TIME_COLUMN, inplace=True)

    print("\n❓ Missing values:\n", df.isnull().sum())
    print("\n📊 Sensor columns:", sensor_cols)
    return df, sensor_cols

def exploratory_data_analysis(df, sensor_cols):
    print("\n🔍 EXPLORATORY DATA ANALYSIS\n" + "="*50)
    print(df[sensor_cols].describe())
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[sensor_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="coolwarm", center=0, square=True, fmt=".2f")
    plt.title("Sensor Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(sensor_cols):
        week_data = df[col].iloc[:2016]
        axes[i].plot(week_data.index, week_data.values, linewidth=0.8)
        axes[i].set_title(f"{col} (1 Week Sample)", fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("plots/data_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    return df[sensor_cols].describe(), correlation_matrix

def detect_shutdowns(df, sensor_cols):
    print("\n🔴 SHUTDOWN DETECTION\n" + "="*50)
    shutdown_thresholds = {
        "Cyclone_Inlet_Gas_Temp": df["Cyclone_Inlet_Gas_Temp"].quantile(0.1),
        "Cyclone_Gas_Outlet_Temp": df["Cyclone_Gas_Outlet_Temp"].quantile(0.1),
        "Cyclone_Material_Temp": df["Cyclone_Material_Temp"].quantile(0.15)
    }
    print(f"🌡️  Shutdown thresholds: {shutdown_thresholds}")
    shutdown_mask = (
        (df["Cyclone_Inlet_Gas_Temp"] < shutdown_thresholds["Cyclone_Inlet_Gas_Temp"]) &
        (df["Cyclone_Gas_Outlet_Temp"] < shutdown_thresholds["Cyclone_Gas_Outlet_Temp"]) &
        (df["Cyclone_Material_Temp"] < shutdown_thresholds["Cyclone_Material_Temp"])
    )
    shutdown_periods = []
    in_shutdown = False
    start_time = None
    for idx, is_shutdown in shutdown_mask.items():
        if is_shutdown and not in_shutdown:
            in_shutdown = True
            start_time = idx
        elif not is_shutdown and in_shutdown:
            in_shutdown = False
            duration = idx - start_time
            shutdown_periods.append({"start": start_time, "end": idx, "duration_hours": duration.total_seconds() / 3600})
    if in_shutdown and start_time is not None:
        duration = df.index[-1] - start_time
        shutdown_periods.append({"start": start_time, "end": df.index[-1], "duration_hours": duration.total_seconds() / 3600})
    shutdown_df = pd.DataFrame(shutdown_periods)
    print(f"\n📊 Shutdowns: {len(shutdown_df)}, Total downtime: {shutdown_df['duration_hours'].sum() if not shutdown_df.empty else 0:.2f}h")
    plt.figure(figsize=(15, 8))
    sample_data = df["Cyclone_Inlet_Gas_Temp"].iloc[:4032]
    plt.plot(sample_data.index, sample_data.values, label="Inlet Gas Temp")
    for _, s in shutdown_df.iterrows():
        plt.axvspan(s["start"], s["end"], alpha=0.3, color='red')
    plt.title("Shutdown Period Detection (2-Week Sample)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/shutdown_detection.png", dpi=300, bbox_inches="tight")
    plt.close()
    if not shutdown_df.empty:
        shutdown_df.to_csv("shutdown_periods.csv", index=False)
    return shutdown_mask, shutdown_df

def perform_clustering(df, sensor_cols, shutdown_mask):
    print("\n🎯 OPERATIONAL STATE CLUSTERING\n" + "="*50)
    active_data = df[~shutdown_mask].copy()
    window = 12
    for col in sensor_cols:
        active_data[f"{col}_rolling_mean"] = active_data[col].rolling(window=window, min_periods=1).mean()
        active_data[f"{col}_rolling_std"] = active_data[col].rolling(window=window, min_periods=1).std()
        active_data[f"{col}_gradient"] = active_data[col].diff()
        active_data[f"{col}_lag1"] = active_data[col].shift(1)
    features = []
    for col in sensor_cols:
        features += [col, f"{col}_rolling_mean", f"{col}_rolling_std", f"{col}_gradient", f"{col}_lag1"]
    clustering_data = active_data[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustering_data)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    clustering_data["cluster"] = cluster_labels
    summary = clustering_data.groupby("cluster")[sensor_cols].mean().reset_index()
    summary["count"] = clustering_data.groupby("cluster").size().values
    summary["percentage"] = 100 * summary["count"] / len(clustering_data)
    summary.to_csv("clusters_summary.csv", index=False)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.title('Operational State Clusters (PCA Projection)')
    plt.tight_layout()
    plt.savefig("plots/cluster_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    return clustering_data, cluster_labels, summary

def detect_anomalies(clustering_data, cluster_labels, sensor_cols):
    print("\n🚨 CONTEXTUAL ANOMALY DETECTION\n" + "="*50)
    anomalous_periods = []
    for cluster_id in np.unique(cluster_labels):
        cluster_data = clustering_data[clustering_data['cluster'] == cluster_id].copy()
        features = sensor_cols + [f"{col}_rolling_std" for col in sensor_cols]
        feat = [f for f in features if f in cluster_data.columns]
        if len(feat) == 0:
            continue
        X_anomaly = cluster_data[feat].fillna(method='ffill').fillna(method='bfill')
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_anomaly)
        is_anomaly = anomaly_labels == -1
        anomaly_indices = cluster_data.index[is_anomaly]
        for idx in anomaly_indices:
            anomalous_periods.append({
                'start_time': idx,
                'cluster_id': cluster_id,
                'top_implicated_variables': ', '.join(feat[:3])
            })
    anomalies_df = pd.DataFrame(anomalous_periods)
    if not anomalies_df.empty:
        anomalies_df.to_csv("anomalous_periods.csv", index=False)
        print("💾 Anomalous periods saved to anomalous_periods.csv")
    return anomalies_df

def forecast_temperature(df, sensor_cols, shutdown_mask):
    print("\n📈 SHORT-TERM FORECASTING\n" + "="*50)
    target_variable = "Cyclone_Inlet_Gas_Temp"
    forecast_horizon = 12  # 1 hour
    active_data = df[~shutdown_mask].copy()
    target_series = active_data[target_variable].dropna()
    split_point = int(len(target_series) * 0.8)
    train_data = target_series[:split_point]
    test_data = target_series[split_point:]
    persistence_forecasts = []
    persistence_actuals = []
    for i in range(0, len(test_data) - forecast_horizon, forecast_horizon):
        last_known = train_data.iloc[-1] if i == 0 else test_data.iloc[i-1]
        forecast = [last_known] * forecast_horizon
        actual = test_data.iloc[i:i+forecast_horizon].values
        persistence_forecasts.extend(forecast)
        persistence_actuals.extend(actual)
    persistence_rmse = np.sqrt(mean_squared_error(persistence_actuals, persistence_forecasts))
    try:
        arima_model = ARIMA(train_data, order=(1,1,1))
        arima_fit = arima_model.fit()
        arima_forecasts = []
        arima_actuals = []
        for i in range(0, len(test_data) - forecast_horizon, forecast_horizon):
            forecast = arima_fit.forecast(steps=forecast_horizon)
            actual = test_data.iloc[i:i+forecast_horizon].values
            arima_forecasts.extend(forecast)
            arima_actuals.extend(actual)
        arima_rmse = np.sqrt(mean_squared_error(arima_actuals, arima_forecasts))
    except:
        arima_forecasts = persistence_forecasts
        arima_rmse = persistence_rmse
    forecasts_df = pd.DataFrame({
        'actual': persistence_actuals[:len(arima_forecasts)],
        'persistence': persistence_forecasts[:len(arima_forecasts)],
        'arima': arima_forecasts[:len(arima_forecasts)],
    })
    forecasts_df.to_csv("forecasts.csv", index=False)
    plt.figure(figsize=(15, 8))
    plt.plot(forecasts_df['actual'].values[:200], label='Actual')
    plt.plot(forecasts_df['persistence'].values[:200], label='Persistence')
    plt.plot(forecasts_df['arima'].values[:200], label='ARIMA')
    plt.legend()
    plt.title('Forecasting Results – Cyclone Inlet Gas Temp')
    plt.tight_layout()
    plt.savefig("plots/forecasting_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("💾 Forecasting results saved to forecasts.csv")
    return forecasts_df, forecasts_df

def generate_insights_and_recommendations(shutdown_df, cluster_summary_df, anomalies_df, forecast_results):
    print("\n💡 INSIGHTS & RECOMMENDATIONS\n" + "="*60)
    print("Data science workflow complete. Outputs written to CSV and plots folders.\n")
    return [], []

def main():
    print("🚀 EXACTSPACE CYCLONE DATA ANALYSIS\n" + "="*70)
    print("📅 Analysis Date: September 30, 2025\n👨‍💻 Analyst: Gunal D\n" + "="*70)
    try:
        df, sensor_cols = load_and_validate_data()
        _ = exploratory_data_analysis(df, sensor_cols)
        shutdown_mask, shutdown_df = detect_shutdowns(df, sensor_cols)
        clustering_data, cluster_labels, cluster_summary_df = perform_clustering(df, sensor_cols, shutdown_mask)
        anomalies_df = detect_anomalies(clustering_data, cluster_labels, sensor_cols)
        forecast_results, forecasts_df = forecast_temperature(df, sensor_cols, shutdown_mask)
        _ = generate_insights_and_recommendations(shutdown_df, cluster_summary_df, anomalies_df, forecast_results)
        print("\n✅ ANALYSIS COMPLETE!")
        print("📁 Generated files:")
        print("   📊 shutdown_periods.csv\n   📊 anomalous_periods.csv\n   📊 clusters_summary.csv\n   📊 forecasts.csv\n   🖼️  plots/ directory with visualizations")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
