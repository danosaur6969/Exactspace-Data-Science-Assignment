***

# Exactspace Data Science Take-Home Assignment

**Author:** Gunal D  
**Email:** [gunalofficialid@gmail.com](mailto:gunalofficialid@gmail.com)  
**Degree:** BTech Computer Science, Bangalore

***

## 🚀 Project Structure

```
Gunal_D_DataScience/
├── Task1/
│   ├── task1_analysis.py
│   ├── README.md
│   ├── shutdown_periods.csv
│   ├── anomalous_periods.csv
│   ├── clusters_summary.csv
│   ├── forecasts.csv
│   └── plots/
│        ├── data_overview.png
│        ├── shutdown_detection.png
│        ├── cluster_analysis.png
│        ├── anomaly_detection.png
│        └── forecasting_results.png
│
├── Task2/
│   ├── architecture_diagram.pptx
│   ├── notes.md
│   └── prototype/
│        ├── rag_prototype.py
│        ├── README.md
│        ├── docs/
│        └── evaluation.csv
│
├── Final_Presentation.pptx
└── CV_Gunal_D.pdf
```

***

## 📂 Task 1: Machine Data Analysis

**Goals:**
- Data cleaning and EDA
- Shutdown detection
- Operational state clustering
- Contextual anomaly detection
- Short-term forecasting
- Summarized actionable insights

**How to Run:**

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly statsmodels hdbscan pyod prophet
   ```
2. Place your `data.xlsx` in the `Task1/` folder.
3. Run:
   ```bash
   python task1_analysis.py
   ```
4. Outputs:
   - Shutdown/anomaly/cluster/forecast results: `.csv` files
   - Key diagnostic plots: `.png` in `plots/`
   - See `Task1/README.md` for details.

***

## 📂 Task 2: RAG + LLM System Prototype

**Features:**
- Document chunking, semantic search (embeddings + FAISS)
- LLM-derived contextual answers with source citation
- Architecture diagram and technical notes included

**How to Run:**

1. Install dependencies:
   ```bash
   pip install sentence-transformers torch transformers faiss-cpu PyPDF2 pdfplumber
   ```
2. Add a few technical PDF docs to `Task2/prototype/docs/`.
3. Run:
   ```bash
   python rag_prototype.py
   ```
4. Outputs:
   - Interactive LLM Q&A and retrieval logs.
   - See `Task2/prototype/README.md` for extra usage details.

***

## 📑 Final Presentation

- `Final_Presentation.pptx` contains summary slides covering approach, workflow, findings, and results for both tasks.

***

## 🧑‍💻 About

This repository is the complete submission for the Exactspace Data Science Take-Home Challenge and serves as a demonstration of:
- Practical machine learning/data analysis skills
- Modern RAG/LLM workflow engineering
- Professional documentation and reproducible project structure

Feel free to clone, explore, and reach out with any questions!

***

**Gunal D**  
[gunalofficialid@gmail.com](mailto:gunalofficialid@gmail.com)

***
