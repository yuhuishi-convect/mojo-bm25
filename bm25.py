import numpy as np
import math
from collections import Counter

# --- BM25 Algorithm Implementation (from previous turn) ---
class BM25:
    """
    A simple implementation of the BM25 ranking algorithm using NumPy.
    """
    def __init__(self, k1=1.5, b=0.75):
        """
        Initializes the BM25 model parameters.

        Args:
            k1 (float): Controls the term frequency saturation. Recommended values are 1.2 to 2.0.
            b (float): Controls the impact of document length normalization. Recommended value is 0.75.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_len = [] # Length of each document
        self.doc_freqs = {} # Number of documents containing a term (Document Frequency)
        self.idf = {} # Inverse Document Frequency for each term
        self.tf = [] # Term Frequency matrix (documents x terms)
        self.vocabulary = [] # List of all unique terms
        self.term_to_id = {} # Mapping from term to its index in vocabulary
        self.bm25_matrix = None # Precomputed BM25 scores for each doc-term

    def fit(self, corpus):
        """
        Trains the BM25 model on the given corpus.
        Calculates document lengths, term frequencies, document frequencies,
        average document length, and IDF values.

        Args:
            corpus (list of list of str): A list where each element is a document,
                                           represented as a list of tokens.
                                           Assumes corpus is pre-processed (e.g., tokenized, lowercased).
        """
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            # print("Warning: Corpus is empty. Cannot fit the model.") # Suppress print in tests
            self.avgdl = 0
            self.doc_len = []
            self.doc_freqs = {}
            self.idf = {}
            self.tf = np.array([])
            self.vocabulary = []
            self.term_to_id = {}
            return

        # 1. Calculate document lengths and build vocabulary
        all_terms = []
        for doc_tokens in corpus:
            self.doc_len.append(len(doc_tokens))
            all_terms.extend(doc_tokens)

        # Calculate average document length
        self.avgdl = np.mean(self.doc_len)

        # Build vocabulary from all unique terms
        unique_terms = sorted(list(set(all_terms)))
        self.vocabulary = unique_terms
        self.term_to_id = {term: idx for idx, term in enumerate(unique_terms)}
        num_terms = len(self.vocabulary)

        if num_terms == 0:
             # print("Warning: Vocabulary is empty. Cannot fit the model.") # Suppress print in tests
             self.doc_freqs = {}
             self.idf = {}
             self.tf = np.zeros((self.corpus_size, 0))
             return


        # 2. Calculate Term Frequencies (TF) and Document Frequencies (DF)
        # Initialize TF matrix: rows are documents, columns are terms in vocabulary
        self.tf = np.zeros((self.corpus_size, num_terms), dtype=np.float32)
        # Initialize Document Frequency counts: how many documents contain each term
        doc_presence = np.zeros((self.corpus_size, num_terms), dtype=bool)

        for i, doc_tokens in enumerate(corpus):
            # Count term frequencies in the current document
            term_counts = Counter(doc_tokens)
            for term, count in term_counts.items():
                if term in self.term_to_id:
                    term_id = self.term_to_id[term]
                    self.tf[i, term_id] = count
                    doc_presence[i, term_id] = True # Mark that this doc contains the term

        # Sum document presence along axis 0 to get document frequencies for each term
        doc_freq_counts = np.sum(doc_presence, axis=0)
        self.doc_freqs = {self.vocabulary[j]: doc_freq_counts[j] for j in range(num_terms)}


        # 3. Calculate IDF for each term
        N = self.corpus_size
        # Using the standard BM25 IDF formula with smoothing (adding 0.5 to numerator and denominator)
        # Avoid division by zero if doc_freqs[term] is N (term in all docs)
        self.idf = {}
        for term in self.vocabulary:
             df = self.doc_freqs[term]
             # Ensure denominator is not zero, although with +0.5 it shouldn't be unless N=0
             if N - df + 0.5 > 0 and df + 0.5 > 0:
                self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
             else:
                # Handle edge case, though log(1)=0 is expected if term is in all docs
                self.idf[term] = 0.0


        # 4. Precompute BM25 matrix (documents x terms)
        doc_len_np = np.array(self.doc_len, dtype=np.float32)
        if self.avgdl == 0:
            length_norm_factor = self.k1 * (1 - self.b)
        else:
            length_norm_factor = self.k1 * (1 - self.b + self.b * doc_len_np / self.avgdl)
        denominator = self.tf + length_norm_factor[:, np.newaxis]
        term_scores_matrix = (self.tf * (self.k1 + 1)) / denominator
        # Multiply each column by the corresponding IDF
        idf_vec = np.array([self.idf[term] for term in self.vocabulary], dtype=np.float32)
        self.bm25_matrix = term_scores_matrix * idf_vec[np.newaxis, :]


    def get_scores(self, query):
        """
        Calculates the BM25 score for each document in the corpus relative to the query.

        Args:
            query (list of str): The query represented as a list of tokens.
                                 Assumes query is pre-processed (e.g., tokenized, lowercased).

        Returns:
            numpy.ndarray: A NumPy array of scores, where scores[i] is the BM25 score
                           of the i-th document with respect to the query.
                           Returns an array of zeros if the model is not fitted or query is empty/out of vocabulary.
        """
        if not hasattr(self, 'bm25_matrix') or self.bm25_matrix is None:
            return np.zeros(self.corpus_size if hasattr(self, 'corpus_size') else 0)

        query_term_indices = [self.term_to_id[term] for term in query if term in self.term_to_id]
        if not query_term_indices:
            return np.zeros(self.corpus_size)
        # Sum the precomputed BM25 scores for the query terms
        scores = np.sum(self.bm25_matrix[:, query_term_indices], axis=1)
        return scores

    def get_top_n(self, query, corpus, n=5):
        """
        Retrieves the top N documents from the corpus based on BM25 scores for the query.

        Args:
            query (list of str): The query represented as a list of tokens.
            corpus (list of list of str): The original corpus (needed to return documents).
            n (int): The number of top documents to return.

        Returns:
            list of tuple: A list of (score, document) tuples for the top N documents,
                           sorted by score in descending order.
                           Returns an empty list if scores cannot be computed or n is 0 or negative.
        """
        if n <= 0:
            return []

        scores = self.get_scores(query)
        if scores is None or scores.shape[0] == 0:
             return [] # Return empty list if scores couldn't be computed or corpus is empty

        # Get the indices of the top n scores in descending order
        # argsort returns indices that would sort the array; [::-1] reverses for descending
        # [:n] takes the top n indices
        # Ensure n doesn't exceed the number of documents
        num_docs_to_return = min(n, self.corpus_size)
        top_n_indices = np.argsort(scores)[::-1][:num_docs_to_return]

        # Retrieve the corresponding documents and scores
        top_documents = [(scores[i], corpus[i]) for i in top_n_indices]

        return top_documents


if __name__ == "__main__":
    test_docs = [
        "The quick brown fox jumps over the lazy dog",
        "Some other text",
        "The quick rabbit runs past the brown fox",
        "The quick rabbit jumps over the brown dog",
        "The quick dog chases past the lazy fox",
        "The quick dog runs through the tall trees",
        "The quick brown fox jumps over the lazy dog",
        "The brown dog sleeps under the shady tree",
        "The brown rabbit hops under the tall tree",
        "The brown fox runs through the forest trees",
        "The brown fox watches the sleeping rabbit",
        "The lazy fox watches over the sleeping dog",
        "The lazy dog watches the quick rabbit",
    ]
    tokenizer = lambda x: x.lower().split()
    test_corps = [tokenizer(doc) for doc in test_docs]

    print(
        f"""
        All documents:
        {test_docs}
        """
    )

    query = input("Enter a query: ")

    query = tokenizer(query)

    model = BM25()
    model.fit(test_corps)

    print(model.get_top_n(query, test_corps))
    