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

import numpy as np
from max.driver import Accelerator, CPU, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from .common import matrix_multiplication


def test_bm25_retrival(session: InferenceSession) -> None:
    num_docs = 100
    num_terms = 20
    tf_matrix = np.random.uniform(size=(num_docs, num_terms)).astype(np.float32)
    query_vector = np.array([
        [1,2,3,4,5],
        [5,4,3,2,1],
    ], dtype=np.int32)

    result = matrix_multiplication(tf_matrix, query_vector, "naive", session, session.devices[0])

    assert result.shape == (2, num_docs)





# def test_optimized_matmul(session: InferenceSession) -> None:
#     M = 256
#     K = 256
#     N = 256

#     a = np.random.uniform(size=(M, K)).astype(np.float32)
#     b = np.random.uniform(size=(K, N)).astype(np.float32)
#     expected_result = a @ b

#     optimized_result = matrix_multiplication(
#         a, b, "optimized", session, session.devices[0]
#     )

#     assert np.all(np.isclose(optimized_result.to_numpy(), expected_result))
#     assert optimized_result.dtype == DType.float32
#     assert optimized_result.shape == (M, N)
