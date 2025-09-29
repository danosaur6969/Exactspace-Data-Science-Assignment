# Task 2: RAG + LLM System Design

## System Architecture Overview

This task designs and prototypes a Retrieval-Augmented Generation (RAG) system for querying technical documents using open-source models.

## Architecture Components

### 1. Document Ingestion & Preprocessing
- PDF parsing and text extraction
- Document metadata extraction
- Content cleaning and normalization

### 2. Chunking Strategy
- Semantic chunking with overlap
- Paragraph-aware splitting
- Section-based hierarchical chunking

### 3. Embeddings & Indexing
- Sentence-transformers embedding model
- FAISS vector database for similarity search
- Hybrid search (dense + sparse retrieval)

### 4. Retrieval Layer
- Multi-stage retrieval pipeline
- Reranking for relevance optimization
- Context-aware chunk selection

### 5. LLM Layer
- Open-source model integration (Hugging Face)
- Prompt engineering for technical domains
- Response generation with citations

### 6. Guardrails & Safety
- Hallucination detection
- Source citation enforcement
- Query filtering and validation

## Installation & Setup

### Prerequisites
```bash
pip install transformers torch sentence-transformers
pip install faiss-cpu chromadb langchain
pip install PyPDF2 pdfplumber streamlit
pip install gradio evaluate datasets
```

### Running the Prototype
```bash
# Start the RAG system
python rag_prototype.py

# Or run with Streamlit interface
streamlit run rag_demo.py
```

## Key Features

### Retrieval Strategy
- **Chunking**: 500-token chunks with 50-token overlap
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Search Method**: Hybrid dense vector + BM25 sparse search
- **Reranking**: Cross-encoder model for relevance scoring

### Guardrails Implementation
- **No Relevant Answers**: Confidence threshold checking
- **Hallucination Prevention**: Source-grounded responses only
- **Citation Enforcement**: Automatic source attribution
- **Query Filtering**: Technical domain validation

### Scalability Design
- **Document Scaling**: Distributed vector indexing
- **User Scaling**: Stateless API design with caching
- **Cost Optimization**: Model quantization and batching

## Output Structure
```
Task2/
├── architecture_diagram.pptx     # Visual system architecture
├── notes.md                      # Design documentation
├── prototype/                    # Working implementation
│   ├── rag_prototype.py         # Main RAG system
│   ├── README.md                # Setup instructions
│   ├── docs/                    # Sample documents
│   │   ├── sample_doc1.pdf
│   │   └── sample_doc2.pdf
│   └── evaluation.csv           # Performance metrics
```

## Technical Implementation

### Document Processing Pipeline
1. PDF text extraction with OCR fallback
2. Document structure analysis
3. Metadata enrichment (title, sections, page numbers)
4. Quality filtering and validation

### Vector Database Setup
- FAISS index with IVF clustering
- Persistent storage for document embeddings
- Metadata filtering capabilities
- Approximate nearest neighbor search

### LLM Integration
- Hugging Face transformers integration
- Model quantization for efficiency
- Batch processing for multiple queries
- Temperature control for consistency

### Evaluation Metrics
- **Precision@K**: Retrieval accuracy
- **Recall@K**: Coverage assessment
- **Response Faithfulness**: Citation accuracy
- **Latency**: End-to-end response time

## Contact
For questions about the implementation, contact: gunal.official@gmail.com