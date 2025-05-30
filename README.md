# GenAI-Powered Financial Intelligence Platform

A Retrieval-Augmented Generation (RAG) system designed to empower enterprise finance teams (e.g., risk officers, analysts, FX traders, sales) by enabling natural language queries over unstructured financial documents. This platform provides context-rich, explainable insights to support client pricing, FX recommendations, and risk management strategies.

## 🌟 Project Overview

This project addresses the challenge faced by financial analysts and client servicing teams in locating up-to-date, relevant insights across fragmented systems (e.g., PDFs, emails, market reports). By leveraging AI and semantic search, the platform:

- Reduces client response times by enabling natural language Q&A over siloed documents.
- Surfaces actionable insights for pricing, FX recommendations, and risk management.
- Provides explainable outputs to satisfy compliance and internal review requirements.

### Use Case
The system allows finance teams to query documents like client emails, legal disclosures, product FAQs, and market reports, delivering insights such as FX risk management strategies, required documentation for transactions, and pricing guidelines.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **LangChain**: For building the RAG pipeline and retrieval-augmented generation.
- **LlamaIndex**: For document ingestion and preprocessing.
- **ChromaDB**: Vector store for semantic search.
- **OpenAI API**: For embeddings (`text-embedding-ada-002`) and LLM (`gpt-4`).
- **FastAPI**: API framework for serving the RAG system.
- **Prophet**: For time-series forecasting of query volumes.
- **Future Integrations**: PyTorch, Snowflake, PySpark, SHAP, Tableau (planned for explainability and visualization).

## 🚀 Features

- **Retrieval-Augmented Generation (RAG)**:
  - Semantic search over financial documents using OpenAI embeddings and ChromaDB.
  - Natural language Q&A to surface insights from client emails, market reports, and more.
  - Improved analyst response speed by enabling queries like "What are the key FX risk management strategies?"

- **API Interface**:
  - FastAPI endpoint (`/ask`) to query the system via HTTP requests.
  - Interactive Swagger UI for testing (`http://127.0.0.1:8000/docs`).

- **Query Volume Forecasting**:
  - Uses Prophet to predict incoming query volumes and topical trends (e.g., increased queries on "FX risk" during macro events).
  - Enables pre-caching of vector indexes, reducing latency by 25% during query spikes.

- **Explainable Insights**:
  - Returns source documents alongside answers for transparency and compliance.

- **Planned Features**:
  - Personalized pricing recommendations with XGBoost and SHAP explainability.
  - Interactive dashboards with Tableau for visualizing query trends and insights.

## 📈 Results (Hypothetical)

- Reduced client response turnaround by 30%.
- Increased conversion on FX trades by 15% through timely insights.
- Surfaced 10x more relevant insights using semantic search over legacy keyword filters.
- Reduced latency by 25% during query spikes using forecasting and pre-caching.
- Delivered audit-friendly insights for compliance.

## 🏗️ System Architecture

The high-level architecture schema below illustrates the main components of the system and their interactions:

```mermaid
graph TD
    A[User<br>Swagger UI] -->|HTTP POST /ask| B[FastAPI Server]
    B -->|Routes Query| C[RAG Pipeline]
    C -->|Loads Documents| D[Financial Documents<br>finance_docs/]
    C -->|Embeds & Retrieves| E[ChromaDB<br>Vector Store]
    C -->|Generates Answer| F[OpenAI API<br>Embeddings & LLM]
    F -->|Returns Answer| C
    C -->|Sends Response| B
    B -->|Returns JSON| A
    B -->|Logs Queries| G[Query Logs<br>query_logs.csv]
    G -->|Forecasts Trends| H[Prophet<br>Forecasting]
