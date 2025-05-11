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
        ],
        custom_extensions=[mojo_kernels],
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

        doc_score = doc_score.permute([2,1,0])

        topk_weight, topk_idx = ops.top_k(doc_score, 1, -1)
        # graph.output(topk_weight)
        graph.output(topk_idx)

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    result = model.execute(a_tensor, b_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    return result.to(CPU())
