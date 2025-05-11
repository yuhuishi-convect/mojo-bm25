# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.engine import InferenceSession


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
        # Check if the model has been fitted and corpus size is valid
        if not hasattr(self, 'corpus_size') or self.corpus_size == 0 or not hasattr(self, 'tf') or not self.idf:
            # print("BM25 model has not been fitted yet or corpus is empty.") # Suppress print in tests
            return np.zeros(self.corpus_size if hasattr(self, 'corpus_size') else 0)

        query_terms = query
        scores = np.zeros(self.corpus_size)

        # Identify query terms present in the vocabulary and get their indices and IDF
        query_term_info = [] # List of (term_id, idf_value) for query terms in vocabulary
        for term in query_terms:
            if term in self.term_to_id:
                term_id = self.term_to_id[term]
                query_term_info.append((term_id, self.idf[term]))

        # If no query terms are in the vocabulary or query is empty, return zero scores
        if not query_term_info:
            # print("None of the query terms are in the vocabulary or query is empty.") # Suppress print in tests
            return scores # Return array of zeros

        # Separate indices and IDF values for vectorized operations
        query_term_indices, query_term_idfs = zip(*query_term_info)
        query_term_indices = list(query_term_indices) # Convert tuple to list for indexing
        query_term_idfs = np.array(query_term_idfs, dtype=np.float32) # Convert to NumPy array

        # TODO: this is t he father
        # Get Term Frequencies for the query terms across all documents
        # tf_subset will have shape (corpus_size, num_query_terms)
        tf_subset = self.tf[:, query_term_indices]

        # Calculate the denominator component that depends on document length
        # This factor is the same for all terms within a single document
        # length_norm_factor will have shape (corpus_size,)
        # Ensure doc_len is a numpy array for element-wise division
        doc_len_np = np.array(self.doc_len, dtype=np.float32)
        # Avoid division by zero if avgdl is zero (only happens with empty corpus, handled above)
        if self.avgdl == 0:
             length_norm_factor = self.k1 * (1 - self.b) # Simplified if avgdl is 0
        else:
             length_norm_factor = self.k1 * (1 - self.b + self.b * doc_len_np / self.avgdl)


        # Apply the BM25 formula using NumPy for vectorization
        # Numerator: tf_subset * (k1 + 1)  (Shape: corpus_size, num_query_terms)
        # Denominator: tf_subset + length_norm_factor[:, np.newaxis]
        # np.newaxis adds a new dimension to length_norm_factor (shape: corpus_size, 1)
        # This allows broadcasting for element-wise addition with tf_subset
        # Avoid division by zero in the denominator
        denominator = tf_subset + length_norm_factor[:, np.newaxis]
        # Handle cases where denominator might be zero (e.g., if tf is -length_norm_factor, which shouldn't happen with non-negative tf and k1, b >= 0)
        # A safer approach might be to add a small epsilon or check for zero
        # Given the formula structure with k1 > 0, the denominator should be > 0 if tf >= 0
        # Let's assume denominator is non-zero under normal conditions.
        term_scores_matrix = (tf_subset * (self.k1 + 1)) / denominator


        # Multiply by IDF and sum across query terms to get final score for each document
        # query_term_idfs shape: (num_query_terms,)
        # term_scores_matrix shape: (corpus_size, num_query_terms)
        # Multiply each column of term_scores_matrix by the corresponding IDF value
        # Using broadcasting: term_scores_matrix * query_term_idfs[np.newaxis, :]
        # weighted_term_scores will have shape (corpus_size, num_query_terms)
        weighted_term_scores = term_scores_matrix * query_term_idfs[np.newaxis, :]

        # Sum along the second axis (axis=1) to get the total score for each document
        # scores will have shape (corpus_size,)
        scores = np.sum(weighted_term_scores, axis=1)

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



def matrix_multiplication(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    algorithm: str,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32
    index_dtype = DType.int32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    a_tensor = Tensor.from_numpy(a).to(device)
    b_tensor = Tensor.from_numpy(b).to(device)

    mojo_kernels = Path(__file__).parent.parent / "operations"

    k1 = 0.5
    b = 0.75
    num_docs = a_tensor.shape[0]
    doc_lengths = np.random.randint(low=1, high=1000, size=num_docs).astype(np.float32)
    avgdl = doc_lengths.mean().item()

    doc_lengths_tensor = Tensor.from_numpy(doc_lengths).to(device)

    # Configure our simple one-operation graph.
    with Graph(
        "bm25_retrieval_graph",
        # tf matrix (num_docs, num_terms) _> (num_docs, num_queries, num_query_terms)
        input_types=[
            TensorType(
                dtype,
                shape=a_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            # query vector (num_queries, num_terms)
            TensorType(
                index_dtype,
                shape=b_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            # doc_lengths (num_docs)
            TensorType(
                dtype,
                shape=doc_lengths.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        tf_matrix, query_vector, doc_lengths = graph.inputs
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.

        weighted_term_scores = ops.gather(
            tf_matrix,
            query_vector,
            axis=1
        )

        # k1 = ops.constant(0.5)
        # b = ops.constant(0.75)

        length_norm_factor = k1 * (1 - b + b * doc_lengths / avgdl)
        denominator = weighted_term_scores + ops.unsqueeze(length_norm_factor, -1)
        term_scores_matrix = (weighted_term_scores * (k1 + 1)) / denominator

        # TODO: handle the query term idf

        # sum over all terms 
        doc_score = ops.sum(
            term_scores_matrix,
            axis=-1
        )

        doc_score = doc_score.transpose(0, -1)

        topk_weight, topk_idx = ops.top_k(doc_score, 1, -1)
        # graph.output(topk_weight)
        graph.output(topk_idx)

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    result = model.execute(a_tensor, b_tensor, doc_lengths_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    return result.to(CPU())
