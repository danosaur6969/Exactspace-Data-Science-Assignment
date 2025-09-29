# Task 1 – Machine Data Analysis (Exactspace Assignment)

## 📌 Overview
This module analyzes **3 years of cyclone machine sensor data** to:
- Detect shutdowns and idle periods
- Segment machine operational states via clustering
- Perform contextual anomaly detection and root cause analysis
- Forecast inlet gas temperature for short horizons
- Deliver actionable engineering insights

---

## 📂 Folder Structure

```
Task1/
├── task1_analysis.py           # Main Python analysis script
├── shutdown_periods.csv        # Shutdown/idle events (start, end, duration)
├── anomalous_periods.csv       # Contextual anomalies with metadata
├── clusters_summary.csv        # Cluster-wise operational stats
├── forecasts.csv               # True vs predicted values for forecasting
├── plots/
│    ├── data_overview.png
│    ├── shutdown_detection.png
│    ├── cluster_analysis.png
│    ├── anomaly_detection.png
│    └── forecasting_results.png
└── README.md                   # This file
```

---

## ⚙️ How to Run

1. **Install Required Packages**  
   Use Python 3.8+ with a virtual environment:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly statsmodels hdbscan pyod prophet
   ```

2. **Place Input Data**  
   Download the provided dataset from Google Drive and save it as:
   ```
   Task1/data.xlsx
   ```

3. **Run the Analysis**  
   Execute directly:
   ```bash
   python task1_analysis.py
   ```
   Or open in Jupyter:
   ```bash
   jupyter notebook task1_analysis.ipynb
   ```

4. **Check Outputs**  
   - `.csv` summaries are saved in the project root  
   - `.png` plots are generated in the `plots/` folder  

---

## 🧠 Methodology

- **Data Preparation**:  
  Cleans missing/error entries (`I/O Timeout`, `Not Connect`) and enforces numeric types.  

- **Exploratory Data Analysis (EDA)**:  
  Visualizes time series, computes summary stats, and correlation matrix.  

- **Shutdown Detection**:  
  Identifies and summarizes idle/shutdown events.  

- **Clustering**:  
  Segments machine states using engineered features (`KMeans`, `HDBSCAN`).  

- **Anomaly Detection**:  
  Finds context-specific anomalies and provides root-cause hypotheses.  

- **Forecasting**:  
  Compares **Persistence**, **ARIMA**, and **Random Forest** for predictive accuracy.  

- **Insights**:  
  Delivers operational recommendations to improve **reliability and efficiency**.  

---

## ❗ Troubleshooting

- **Error loading data** → Ensure the file is named `data.xlsx` and has correct columns.  
- **Missing packages** → Reinstall using the pip command above.  
- **Type conversion issues** → Update error string list in the script.  

---

## 👤 Author

**Gunal D**  
📧 [gunalofficialid@gmail.com](mailto:gunalofficialid@gmail.com)  
🎓 BTech Computer Science Engineering, Bangalore  
