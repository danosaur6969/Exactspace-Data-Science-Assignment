#!/usr/bin/env python3
"""
RAG + LLM System Prototype for Technical Document Query
Author: Gunal D
Date: September 30, 2025
Assignment: Exactspace Data Science Take-Home Challenge - Task 2

This script implements a Retrieval-Augmented Generation (RAG) system
for querying technical cyclone separator documentation.
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path

# PDF processing
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("⚠️ PDF libraries not available. Install with: pip install PyPDF2 pdfplumber")

# ML/NLP libraries
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import faiss
except ImportError:
    print("⚠️ ML libraries not available. Install with: pip install sentence-transformers torch transformers faiss-cpu")

# Optional libraries
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
@dataclass
class QueryResult:
    """Represents a query result with retrieved chunks and generated answer"""
    query: str
    answer: str
    retrieved_chunks: List[DocumentChunk]
    confidence_score: float
    sources: List[str]

class DocumentProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF"""
        text_content = ""
        metadata = {"source": pdf_path, "pages": 0, "title": ""}
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                metadata["title"] = os.path.basename(pdf_path).replace('.pdf', '')
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
                        
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(reader.pages)
                    
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n[Page {page_num + 1}]\n{page_text}\n"
                            
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e2}")
                return "", metadata
        
        return text_content, metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page break artifacts
        text = re.sub(r'\[Page \d+\]\s*\[Page \d+\]', '[Page Break]', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Clean up special characters
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        return text.strip()

class ChunkingStrategy:
    """Implements various text chunking strategies"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def semantic_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk text based on semantic boundaries"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_type": "semantic",
                    "char_count": len(current_chunk)
                })
                
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata
                ))
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
                    
                chunk_id += 1
            else:
                current_chunk += " " + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_type": "semantic",
                "char_count": len(current_chunk)
            })
            
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def fixed_size_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Simple fixed-size chunking with overlap"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + self.chunk_size // 2:
                    chunk_text = chunk_text[:boundary + 1]
                    end = start + len(chunk_text)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_type": "fixed_size",
                "char_count": len(chunk_text),
                "start_pos": start,
                "end_pos": end
            })
            
            chunks.append(DocumentChunk(
                content=chunk_text.strip(),
                metadata=chunk_metadata
            ))
            
            start = end - self.overlap
            chunk_id += 1
        
        return chunks

class VectorStore:
    """Manages document embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to vector store"""
        if not self.embedding_model:
            logger.error("No embedding model available")
            return
            
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        # Create FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add to index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        self.chunk_metadata.extend([chunk.metadata for chunk in chunks])
        
        logger.info(f"Vector store now contains {len(self.chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if not self.embedding_model or not self.index:
            logger.error("Vector store not properly initialized")
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(sim)))
                
        return results

class LLMGenerator:
    """Handles LLM-based answer generation"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        try:
            # Use a lightweight model for demonstration
            # In production, consider using Llama-2 7B or similar
            self.generator = pipeline(
                "text-generation",
                model="gpt2",  # Fallback to GPT-2 for compatibility
                tokenizer="gpt2",
                max_length=1000,
                device=-1  # Use CPU
            )
            logger.info(f"Loaded LLM: gpt2")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.generator = None
    
    def generate_answer(self, query: str, contexts: List[str], max_length: int = 500) -> Tuple[str, float]:
        """Generate answer based on query and retrieved contexts"""
        if not self.generator:
            return "LLM not available for answer generation.", 0.0
        
        # Construct prompt
        context_str = "\n\n".join(contexts[:3])  # Use top 3 contexts
        
        prompt = f"""Based on the following technical documentation, answer the question accurately and cite your sources.

Context:
{context_str}

Question: {query}

Answer:"""
        
        try:
            # Generate response
            outputs = self.generator(
                prompt,
                max_length=len(prompt) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract answer (text after "Answer:")
            answer_start = generated_text.find("Answer:") + 7
            answer = generated_text[answer_start:].strip()
            
            # Simple confidence scoring based on length and context overlap
            confidence = min(0.9, len(answer) / max_length + 0.3)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Unable to generate answer: {str(e)}", 0.0

class RAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.doc_processor = DocumentProcessor()
        self.chunking_strategy = ChunkingStrategy(chunk_size, overlap)
        self.vector_store = VectorStore()
        self.llm_generator = LLMGenerator()
        self.confidence_threshold = 0.3
        
    def ingest_documents(self, doc_directory: str) -> None:
        """Ingest all PDF documents from directory"""
        doc_path = Path(doc_directory)
        if not doc_path.exists():
            logger.error(f"Document directory not found: {doc_directory}")
            return
            
        pdf_files = list(doc_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text
            text, metadata = self.doc_processor.extract_text_from_pdf(str(pdf_file))
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue
                
            # Clean text
            clean_text = self.doc_processor.clean_text(text)
            
            # Create chunks
            chunks = self.chunking_strategy.semantic_chunking(clean_text, metadata)
            all_chunks.extend(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
        
        # Add to vector store
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            logger.info(f"Successfully ingested {len(all_chunks)} chunks")
        else:
            logger.warning("No chunks were created from documents")
    
    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Process a query and return results"""
        logger.info(f"Processing query: {question}")
        
        # Input validation
        if not question.strip():
            return QueryResult(
                query=question,
                answer="Please provide a valid question.",
                retrieved_chunks=[],
                confidence_score=0.0,
                sources=[]
            )
        
        # Retrieve relevant chunks
        search_results = self.vector_store.search(question, k=top_k)
        
        if not search_results:
            return QueryResult(
                query=question,
                answer="I couldn't find any relevant information in the documents to answer your question.",
                retrieved_chunks=[],
                confidence_score=0.0,
                sources=[]
            )
        
        # Filter by confidence threshold
        filtered_results = [(chunk, score) for chunk, score in search_results if score >= self.confidence_threshold]
        
        if not filtered_results:
            return QueryResult(
                query=question,
                answer="I found some potentially related information, but I'm not confident it answers your question accurately.",
                retrieved_chunks=[chunk for chunk, _ in search_results],
                confidence_score=0.0,
                sources=[]
            )
        
        # Prepare contexts for LLM
        contexts = [chunk.content for chunk, _ in filtered_results]
        chunks = [chunk for chunk, _ in filtered_results]
        
        # Generate answer
        answer, confidence = self.llm_generator.generate_answer(question, contexts)
        
        # Extract sources
        sources = list(set([chunk.metadata.get('source', 'Unknown') for chunk in chunks]))
        
        # Add citations to answer
        if sources and answer != "LLM not available for answer generation.":
            citations = ", ".join([os.path.basename(src) for src in sources])
            answer += f"\n\nSources: {citations}"
        
        return QueryResult(
            query=question,
            answer=answer,
            retrieved_chunks=chunks,
            confidence_score=confidence,
            sources=sources
        )
    
    def evaluate_retrieval(self, test_queries: List[Dict]) -> pd.DataFrame:
        """Evaluate retrieval performance"""
        results = []
        
        for test_case in test_queries:
            query = test_case['query']
            expected_docs = test_case.get('expected_docs', [])
            
            search_results = self.vector_store.search(query, k=10)
            retrieved_docs = [chunk.metadata['source'] for chunk, _ in search_results]
            
            # Calculate metrics
            relevant_retrieved = len(set(expected_docs) & set(retrieved_docs))
            precision_at_5 = relevant_retrieved / min(5, len(retrieved_docs))
            recall = relevant_retrieved / len(expected_docs) if expected_docs else 0
            
            results.append({
                'query': query,
                'precision@5': precision_at_5,
                'recall': recall,
                'retrieved_count': len(retrieved_docs)
            })
        
        return pd.DataFrame(results)

def create_sample_documents():
    """Create sample documents for demonstration"""
    os.makedirs("docs", exist_ok=True)
    
    # Sample cyclone documentation content
    sample_docs = [
        {
            "filename": "docs/cyclone_operation_manual.txt",
            "content": """
Cyclone Separator Operation Manual

1. Introduction
Cyclone separators are devices used to remove particulates from an air, gas or liquid stream, without the use of filters, through vortex separation.

2. Operating Parameters
- Inlet Gas Temperature: Normal range 400-500°C
- Outlet Gas Temperature: Typically 50-80°C lower than inlet
- Draft Pressure: Maintains negative pressure for proper flow
- Material Temperature: Should remain below 250°C for safety

3. Shutdown Procedures
When inlet gas temperature drops below 300°C or draft pressure becomes positive, initiate shutdown sequence:
1. Reduce feed rate gradually
2. Monitor temperature decline
3. Secure all valves when temperature reaches ambient

4. Troubleshooting
- High outlet temperature: Check for blockages or reduced efficiency
- Pressure anomalies: Inspect draft fans and ductwork
- Material temperature spikes: Reduce feed rate, check cooling systems
            """
        },
        {
            "filename": "docs/cyclone_maintenance_guide.txt", 
            "content": """
Cyclone Separator Maintenance Guide

1. Routine Maintenance
- Daily temperature and pressure logging
- Weekly visual inspections
- Monthly cleaning cycles

2. Anomaly Response
Sudden temperature drops often indicate:
- Upstream feed interruption
- Draft fan malfunction
- Blockage clearing events

Temperature spikes may signal:
- Excessive feed rate
- Cooling system failure
- Material composition changes

3. Preventive Measures
- Install temperature gradient monitoring
- Implement predictive maintenance based on operational patterns
- Maintain spare parts inventory for critical components

4. Safety Procedures
- Never exceed maximum operating temperatures
- Ensure proper ventilation during maintenance
- Use appropriate PPE when inspecting hot surfaces
            """
        }
    ]
    
    # Write sample documents
    for doc in sample_docs:
        with open(doc["filename"], 'w') as f:
            f.write(doc["content"])
    
    print("✅ Sample documents created in docs/ directory")

def main():
    """Main execution function for RAG prototype"""
    print("🚀 RAG + LLM SYSTEM PROTOTYPE")
    print("=" * 50)
    print("📅 Date: September 30, 2025")
    print("👨‍💻 Developer: Gunal D")
    print("=" * 50)
    
    # Create sample documents if docs directory doesn't exist
    if not os.path.exists("docs") or not os.listdir("docs"):
        print("📄 Creating sample documents...")
        create_sample_documents()
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    rag_system.ingest_documents("docs")
    
    # Test queries
    test_queries = [
        "What does a sudden draft drop indicate?",
        "What is the normal operating temperature range for cyclone inlet?",
        "How should I respond to temperature spikes?",
        "What are the shutdown procedures for a cyclone?",
        "What causes high outlet temperatures?"
    ]
    
    print("\n🔍 TESTING RAG SYSTEM")
    print("=" * 50)
    
    results = []
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = rag_system.query(query)
        
        print(f"✅ Answer: {result.answer}")
        print(f"📊 Confidence: {result.confidence_score:.2f}")
        print(f"📚 Sources: {len(result.sources)} documents")
        print(f"🔍 Retrieved: {len(result.retrieved_chunks)} chunks")
        print("-" * 50)
        
        results.append({
            'query': query,
            'answer': result.answer,
            'confidence': result.confidence_score,
            'source_count': len(result.sources),
            'chunk_count': len(result.retrieved_chunks)
        })
    
    # Save evaluation results
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation.csv", index=False)
    print("💾 Evaluation results saved to evaluation.csv")
    
    # Interactive mode
    print("\n🎯 INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_query:
                continue
                
            result = rag_system.query(user_query)
            print(f"\n🤖 Answer: {result.answer}")
            print(f"📊 Confidence: {result.confidence_score:.2f}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n✅ RAG System Demo Complete!")
    print("📁 Generated files:")
    print("   📄 docs/cyclone_operation_manual.txt")
    print("   📄 docs/cyclone_maintenance_guide.txt") 
    print("   📊 evaluation.csv")

if __name__ == "__main__":
    main()