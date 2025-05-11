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
from typing import Tuple
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.engine import InferenceSession

    



def gpu_execute_query(
    score_matrix: NDArray[np.float32],
    query_vector: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Tuple[Tensor, Tensor]:
    dtype = DType.float32
    index_dtype = DType.int32
    # execute the model
    breakpoint()
    score_matrix = Tensor.from_numpy(score_matrix).to(device)
    query_vector = Tensor.from_numpy(query_vector).to(device)
    with Graph(
        "bm25_retrieval_graph",
        # tf matrix (num_docs, num_terms) _> (num_docs, num_queries, num_query_terms)
        input_types=[
            TensorType(
                dtype,
                shape=score_matrix.shape,
                device=DeviceRef.from_device(device),
            ),
            # query vector (num_query_terms, )
            TensorType(
                index_dtype,
                shape=query_vector.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        # custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        tf_matrix, query_vector = graph.inputs
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.

        weighted_term_scores = ops.gather(
            tf_matrix,
            query_vector,
            axis=1
        )

        # sum over all terms 
        doc_score = ops.sum(
            weighted_term_scores,
            axis=-1
        )

        doc_score = doc_score.transpose(0, -1)

        topk_weight, topk_idx = ops.top_k(doc_score, 1, -1)
        # graph.output(topk_weight)
        graph.output(topk_idx, topk_weight)

    model = session.load(graph)

    topk_idx, topk_weight = model.execute(score_matrix, query_vector)
    return topk_idx.to(CPU()), topk_weight.to(CPU())
    
