General-purpose medical AI presents core challenges including hallucination risks, traceability gaps, and safety/diagnostic distinctions. Clinic GPT solves these challenges by implementing a grounded RAG pipeline that achieves 0% outside knowledge and 100% traceable claims.
Mitigates Hallucination Risk: Prevents general-purpose LLMs from generating unverified clinical advice
Ensures Traceability: Provides direct, verifiable citation mechanisms linking answers to specific pages and text chunks
Enforces Safety & Triage: Clearly distinguishes routine screening tools from formal clinical diagnoses.

---------------------

Architecture & Pipeline
Clinic GPT follows a three-layer pipeline architecture:  
**Data Preparation Layer**: 
Source corpus: 2025 ESC Guidelines PDF. 
Loaders and splitters: PyPDFLoader and RecursiveCharacterTextSplitter 
(configured with a chunk size of 850 and overlap of 150). 

Retrieval Pipeline Layer:
Embeddings: SentenceTransformer (all-MiniLM-L6-v2, 384-d). 
Vector Index: FAISS (Normalized $L2/IP$

Generation & Safeguard Layer:Safety Intercept Gate: Crisis & medication filters. 
Context Builder and LLM Engine: Groq / openai/gpt-oss-120b producing cited clinical answers

Knowledge Base & Clinical Scope
Target Corpus: 2025 ESC Guidelines (Management of Cardiovascular Disease and Mental Health Disorders, 70 PDF pages, 693 baseline chunks
Depression & Major Depressive Disorder  Generalized Anxiety & Panic Disorders  Post-Traumatic Stress Disorder (PTSD)  General Psychological Distress in CVD  Key Screening Tools: PHQ-2, PHQ-9, GAD-2, GAD-7, and Whooley Questions. 
Intervention Frameworks: Psychological, lifestyle, cardiac rehabilitation, and pharmacological safety considerations.


Safety by Design & Triage Pipeline.
The system uses a pattern-based safety gate on incoming user questions to route queries appropriately:
Crisis Intercept: Detects crisis patterns and directs users immediately to national crisis lifelines without executing RAG generation
Medication Guardrail: Identifies medication patterns and redirects drug dosing or alteration requests to formal clinical consultations.
Grounded RAG Route: Processes normal clinical queries to generate evidence-backed summaries with explicit inline citations

Clinical Triage Distinction: Screening tools (PHQ-2, GAD-2, Whooley) are strictly restricted to identifying symptom levels and do not provide a clinical diagnosis
