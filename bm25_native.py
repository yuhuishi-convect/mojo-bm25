from enum import Enum
from typing import Dict, Generic, List, Optional, Tuple, TypeAlias, TypeVar
import logging

import scipy.sparse as sp
import numpy as np
import numpy.typing as npt

QueryId: TypeAlias = np.int32
TokId: TypeAlias = np.int32
DocId: TypeAlias = np.int32
Score: TypeAlias = np.float32


SpType = TypeVar("SpType")
Shape = TypeVar("Shape")
DType = TypeVar("DType")


# https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
class NPArray(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    pass


class BM25v:
    """
    BM25v(ector)

    BM25 implementation designed to work with pre-computed sparse matrices in CSC format.
    This class stores document-token matrices as CSC matrices and provides efficient BM25 scoring.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25v index.

        Args:
            k1: Term (token) saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.dtype = np.float32

        self.doc_toks: sp.csc_matrix = sp.csc_matrix(np.zeros((0,), dtype=self.dtype))
        self.doc_lengths: np.ndarray = np.zeros((0,), dtype=self.dtype)
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def index(
        self, doc_toks: sp.csc_matrix, doc_lengths: npt.NDArray[np.int32]
    ) -> None:
        """
        Index the document-token matrix.
        Each row is a document and each column is a token.
        The values are the pre-scored BM25 scores for each token in each document.

        Args:
            doc_toks: Sparse document-token matrix in CSC format, i.e. scores matrix
            doc_lengths: Document lengths (number of terms in each document).
        """
        self.doc_toks = doc_toks
        self.doc_lengths = doc_lengths
        self.avg_doc_length = np.mean(doc_lengths)
        self.num_docs = doc_toks.shape[0]

    def search(
        self, queries: NPArray["Q,T", np.int32], top_k: int = 100
    ) -> Tuple[NPArray["D", np.int32], NPArray["D", np.float32]]:
        """
        Search for the top-k documents matching the query.

        Args:
        query_mat: ndarray (matrix) of queries, each element is an array of token IDs.
            top_k: Number of top documents to return.

        Returns:
            Tuple of sorted ndarrays of top k document IDs and their scores.
        """
        if self.num_docs is None:
            raise ValueError("BM25v index not built. Call index() first.")

        if len(queries) == 0:
            self.logger.info(
                msg="The query is empty. This will result in a zero score for all documents."
            )
            docs, scores = (
                np.zeros((0, 0), dtype=self.dtype),
                np.zeros((0, 0), dtype=self.dtype),
            )
        else:
            docs, scores = self.get_scores(queries, top_k)

        return docs, scores

    def get_scores(
        self, queries: NPArray["Q,T", TokId], top_k: int
    ) -> Tuple[NPArray["Q,D", np.float32], NPArray["Q,D", np.float32]]:
        if (
            not isinstance(queries, np.ndarray)
            or queries.ndim != 2
            or not isinstance(queries[0][0], TokId)
        ):
            breakpoint()
            raise ValueError("The queries must be a list of list of query token IDs.")

        max_token_id = int(queries.max(initial=0))

        if max_token_id >= len(self.doc_toks.indptr) - 1:
            raise ValueError(
                f"The maximum token ID in the query ({max_token_id}) is higher than the number of tokens in the index."
            )

        return self._compute_relevance_from_scores(
            queries=queries,
            top_k=top_k,
            dtype=self.dtype,
        )

    def _compute_relevance_from_scores(
        self,
        queries: NPArray["Q,T", np.int32],
        top_k: int,
        dtype: np.dtype,
    ) -> Tuple[NPArray["Q,D", np.float32], NPArray["Q,D", np.float32]]:
        """
        Compute BM25 relevance scores for the given query tokens.

        Args:
            query_tokens_ids: Query token IDs, each row is a query.
            top_k: Number of top documents to return.
            dtype: Data type for the output scores.

        Returns:
            Tuple of sorted ndarrays of top k document IDs and their scores.
        """

        top_docs = np.zeros((queries.shape[0], top_k), dtype=DocId)
        top_scores = np.zeros((queries.shape[0], top_k), dtype=Score)
        for i in range(len(queries)):
            query = queries[i]
            query = query[query >= 0]
            doc_scores = self.doc_toks[:, query].sum(axis=1).A1

            top_docs_i, top_scores_i = _topk(doc_scores, top_k)

            top_docs[i] = top_docs_i
            top_scores[i] = top_scores_i
        return top_docs, top_scores

    def _compute_relevance_from_scores_matmul(
        self,
        queries: NPArray["Q,T", np.int32],
        top_k: int,
        dtype: np.dtype,
    ) -> Tuple[NPArray["Q,D", np.float32], NPArray["Q,D", np.float32]]:
        """
        Compute BM25 relevance scores for the given query tokens, using matrix multiplication.
        """

        top_docs = np.zeros((queries.shape[0], top_k), dtype=DocId)
        top_scores = np.zeros((queries.shape[0], top_k), dtype=Score)

        # construct sparse matrix for queries
        q_cols = np.array([j for i, xs in enumerate(queries) for j in [i] * len(xs)])
        q_rows = np.array(np.concatenate(queries))
        q_data = np.ones(len(q_rows), dtype=np.float32)
        q_shape = (self.doc_toks.shape[1], len(queries))
        qsp = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=q_shape)

        scores_all = self.doc_toks.dot(qsp).transpose()

        doc_scores_all = scores_all.toarray()

        for i in range(len(queries)):
            doc_scores = doc_scores_all[i, :]

            top_docs_i, top_scores_i = _topk(doc_scores, top_k)

            top_docs[i] = top_docs_i
            top_scores[i] = top_scores_i

        return top_docs, top_scores


def _topk_sort(doc_scores, k):
    """
    Slower version without argpartition.
    """
    top_docs_i = np.flip(np.argsort(doc_scores)[-k:])
    top_scores_i = np.take(doc_scores, top_docs_i)
    return top_docs_i, top_scores_i


def _topk(doc_scores, k):
    docs_i = np.argpartition(doc_scores, -k)
    top_docs_i = docs_i.take(indices=range(-k, 0))
    top_scores_i = np.take(doc_scores, top_docs_i)

    sorted_trunc_ind = np.flip(np.argsort(top_scores_i))

    top_docs_i = top_docs_i[sorted_trunc_ind]
    top_scores_i = top_scores_i[sorted_trunc_ind]

    return top_docs_i, top_scores_i




if __name__ == "__main__":
    test_corps = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",   
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]

    tokenizer = lambda x: x.lower().split()
    
    corpus_tokens = [tokenizer(doc) for doc in test_corps]
    doc_lengths = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)

    doc_scores = np.array([[1.0,2.0,3.0],
                            [2.0,4.0,1.0]], dtype=np.float32)
    doc_matrix = sp.csc_matrix(doc_scores, (len(doc_scores), len(doc_scores[0])), dtype=np.float32)
 
    
    model = BM25v()
    
    model.index(doc_matrix, np.array([2], dtype=np.int32))
    
    query = np.array([[0,1]], dtype=np.int32)
    
    docs, scores = model.search(query, top_k=1)
    
    print(docs)
    print(scores)
    
    print(model.doc_toks.toarray())

    
