🌿 BRSR Principle 6 Faithfulness Audit (Infosys)
This repository provides a fully-automated Python workflow to conduct a faithfulness audit of a company's Business Responsibility and Sustainability Report (BRSR) against the SEBI regulatory requirements, focusing on Principle 6 (Environment).

The core comparison is performed using a Retrieval-Augmented Generation (RAG) approach powered by the Gemini API, ensuring the analysis is grounded in the source documents.

✨ Features
PDF Extraction: Downloads and extracts text from both the official SEBI BRSR Guideline and the Infosys BRSR Report (2022-23).

AI-Powered RAG: Utilizes the Gemini API for semantic text chunking, embedding, vector indexing, and RAG-style comparison.

Compliance Scoring: Computes a quantitative faithfulness drift score (0-3) for predefined Principle 6 concepts, complete with textual explanations and direct source citations.

Visualization & Reporting: Generates clear, actionable deliverables including an audit table, an interactive Sankey diagram, and a color-coded HTML dashboard.

Flexible Embeddings: Supports both proprietary embeddings (--embedder openai) and privacy-focused local embeddings (--embedder local).