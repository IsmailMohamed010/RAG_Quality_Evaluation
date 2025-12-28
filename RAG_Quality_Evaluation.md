# RAG Quality Evaluation 

## Scenario Setup
**Example Document:** Technical FAQ about "Cloud Storage Solutions"
**Sample Questions:**
1. "How do I recover deleted files?"
2. "What security features are included?"
3. "What are the pricing tiers?"

---

## 1. CHUNKING STRATEGY

### Recommended Approach: **Semantic Chunking with Overlap**

**Strategy:**
- **Chunk size:** 512 tokens (~2000 characters) per chunk
- **Overlap:** 100 tokens (20% overlap) to preserve context at boundaries
- **Boundary detection:** Split on semantic units (sections, paragraphs) rather than mid-sentence
- **Method:** Recursive text splitting

**Why this works:**
- 512 tokens fits within most embedding models' context windows
- Overlap preserves cross-boundary context (important for questions spanning multiple ideas)
- Semantic boundaries maintain coherence (don't split paragraphs arbitrarily)

**Pseudocode:**
```
chunks = []
for section in document.sections:
    text = section.text
    tokens = tokenize(text)
    
    for i in range(0, len(tokens), stride=412):  # 512 - 100 overlap
        chunk_tokens = tokens[i : i+512]
        chunks.append(detokenize(chunk_tokens))
        
return chunks
```

---

## 2. EMBEDDINGS & RETRIEVAL (Pseudocode)

### Step 1: Generate Embeddings
```python
# Initialization
embedding_model = load_model("sentence-transformers/all-MiniLM-L6-v2")
vector_store = initialize_vector_db("FAISS")  # or Pinecone, Weaviate

# Embed all chunks
for chunk in chunks:
    embedding = embedding_model.encode(chunk)  # 384-dim vector
    vector_store.add(chunk_id, embedding, metadata={"text": chunk})
```

### Step 2: Retrieve for Query
```python
def retrieve(query, top_k=2):
    # Embed user query with SAME model
    query_embedding = embedding_model.encode(query)
    
    # Vector similarity search
    results = vector_store.search(
        query_embedding, 
        top_k=top_k,
        metric="cosine_similarity"
    )
    
    # Rerank (optional but recommended)
    reranked = rerank_model.rank(query, results)
    
    return reranked[:top_k]

# Example usage
results = retrieve("How do I recover deleted files?", top_k=2)
```

### Step 3: Use in Generation
```python
def rag_response(query):
    # Retrieve
    relevant_chunks = retrieve(query)
    
    # Build context
    context = "\n---\n".join([r.text for r in relevant_chunks])
    
    # Generate with LLM
    prompt = f"""Context: {context}
    
Question: {query}

Answer:"""
    
    response = llm.generate(prompt)
    return response
```

---

## 3. RELEVANCE SELECTION (Top 1-2 Chunks per Question)

### Question 1: "How do I recover deleted files?"
**Selected Chunks:**
- **Chunk A:** "Deleted files enter the trash bin for 30 days before permanent deletion. Users can restore from trash, or contact support for recovery within 90 days using backup snapshots..."
- **Chunk B:** "Version control is available on all plans. Each file stores up to 10 previous versions, allowing rollback to any previous state..."

### Question 2: "What security features are included?"
**Selected Chunks:**
- **Chunk C:** "All plans include AES-256 encryption at rest, TLS 1.2+ for transit, and two-factor authentication. Enterprise plans add role-based access control and audit logging..."
- **Chunk D:** "Compliance certifications: SOC 2 Type II, ISO 27001, HIPAA, GDPR..."

### Question 3: "What are the pricing tiers?"
**Selected Chunks:**
- **Chunk E:** "Three tiers: Basic ($9.99/mo, 100GB), Professional ($19.99/mo, 1TB), Enterprise (custom pricing)..."
- **Chunk F:** "All plans include 24/7 customer support, except Basic which has email-only support..."

---

## 4. WHY THESE CHUNKS ARE MOST RELEVANT

### Relevance Scoring Method:
1. **Keyword overlap:** "deleted/recovery", "security features", "pricing" appear in both query and chunk
2. **Semantic similarity:** Query embedding has high cosine similarity (>0.75) to chunk embedding
3. **Chunk proximity:** Related chunks appear near retrieved chunk in vector space
4. **Specificity:** Chunks answer the exact question (not generic info)

### Explanation by Question:

**Q1 - Recovery:** Chunks A & B directly address deletion scenarios and version history—directly answering "how to recover"

**Q2 - Security:** Chunks C & D exhaustively cover security features (encryption, 2FA, compliance)—exactly what the query asks

**Q3 - Pricing:** Chunks E & F cover pricing tiers and included features—complete answer to pricing question

**Why NOT other chunks?** Generic sections (company history, feature overview) lack direct relevance and would add noise.

---

## 5. THREE COMMON RAG ISSUES & FIXES

### Issue 1: Retrieval Failure (Wrong/Missing Context)
**Problem:** Chunks retrieved are irrelevant; LLM hallucinates or gives incorrect answers
**Root causes:**
- Poorly chunked documents (context split mid-idea)
- Embedding model misalignment (query uses synonyms, model didn't train on them)
- Low similarity threshold filters good chunks

**Fixes:**
1. **Improve chunking:** Use semantic boundaries, add metadata tags (section headers, doc type)
2. **Rerank retrieved chunks:** Use cross-encoder reranker to re-score BM25 + vector results
3. **Hybrid retrieval:** Combine vector search (semantic) with BM25 (keyword matching)
4. **Fine-tune embeddings:** Train embeddings on domain-specific data

**Implementation:**
```python
# Hybrid approach
def hybrid_retrieve(query, top_k=5):
    # Vector search
    vector_results = vector_store.search(query_embedding, top_k=10)
    
    # BM25 keyword search
    bm25_results = bm25_index.search(query, top_k=10)
    
    # Merge and deduplicate
    combined = merge_results(vector_results, bm25_results)
    
    # Rerank
    reranked = rerank_model.rank(query, combined)
    return reranked[:top_k]
```

---

### Issue 2: Token Limit Overflows
**Problem:** Context too large; exceeds LLM token limit or forces truncation of chunks
**Root causes:**
- Too many large chunks retrieved
- Chunk overlap inflates size
- No context compression

**Fixes:**
1. **Reduce retrieved chunks:** Retrieve only top 2-3 chunks (not top 10)
2. **Compress context:** Summarize chunks or extract key sentences
3. **Increase context window:** Use models with larger windows (GPT-4, Claude 3)
4. **Smart chunk size:** Smaller chunks (256 tokens) for long documents

**Implementation:**
```python
def rag_with_compression(query):
    chunks = retrieve(query, top_k=2)
    
    # Compress each chunk
    compressed = [compress_chunk(c) for c in chunks]
    
    # Verify token count
    tokens = count_tokens(compressed)
    if tokens > max_tokens:
        compressed = [c[:100] for c in compressed]  # Truncate
    
    return generate_answer(compressed)
```

---

### Issue 3: Hallucination & Inconsistency
**Problem:** LLM generates plausible-sounding but false answers not in retrieved chunks
**Root causes:**
- Retrieved chunks don't contain complete answer
- LLM uses pre-training knowledge instead of context
- No consistency checks between retrieval and generation

**Fixes:**
1. **Force citation:** Prompt LLM to cite source chunks; validate citations exist
2. **Temperature control:** Set temperature=0 for deterministic, factual answers
3. **Grounding check:** Verify generated answer's key claims appear in retrieved chunks
4. **Chain-of-thought:** Ask LLM to explain which chunk each fact comes from

**Implementation:**
```python
def grounded_rag(query):
    chunks = retrieve(query)
    
    prompt = f"""Use ONLY these chunks to answer. Cite sources.
    
Chunks:
{chunks_text}

Question: {query}

Answer (with citations):"""
    
    response = llm.generate(prompt, temperature=0)
    
    # Validate: check if key claims cite sources
    validate_citations(response, chunks)
    
    return response
```
