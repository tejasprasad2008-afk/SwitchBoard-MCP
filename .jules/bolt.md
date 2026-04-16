## 2025-05-15 - [NumPy batch operations for semantic similarity]
**Learning:** Using NumPy matrix-vector multiplication for computing cosine similarity against multiple category embeddings is significantly faster than looping and computing them individually in Python, especially when the number of categories or the dimensionality of embeddings is high. Pre-normalizing the category matrix avoids redundant norm calculations.
**Action:** Always prefer NumPy batch operations for similarity searches or classifications against a fixed set of reference embeddings.
