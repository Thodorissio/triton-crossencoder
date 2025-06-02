import time
import numpy as np
import tritonclient.grpc as grpcclient
from copy import deepcopy
from langchain_core.documents import Document
from langchain_community.cross_encoders import BaseCrossEncoder


class TritonCrossEncoder(BaseCrossEncoder):
    """Trtion Cross Encoder for reranking."""

    def __init__(
        self,
        triton_url: str,
        model_name: str = "bge-reranker-v2-m3",
        max_batch_size: int = 8,
        input_name: str = "INPUT_PAIRS",
        output_name: str = "SCORES",
    ):
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
        self._max_batch_size = max_batch_size
        self._input_name = input_name
        self._output_name = output_name

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Score text pairs using the Triton model."""

        all_scores = []
        for i in range(0, len(text_pairs), self._max_batch_size):
            batch = text_pairs[i : i + self._max_batch_size]

            np_input = np.array(
                [[pair[0].encode("utf-8"), pair[1].encode("utf-8")] for pair in batch],
                dtype=object,
            )
            infer_input = grpcclient.InferInput(
                self._input_name, np_input.shape, "BYTES"
            )
            infer_input.set_data_from_numpy(np_input)

            infer_output = grpcclient.InferRequestedOutput(self._output_name)
            results = self.triton_client.infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[infer_output],
            )
            scores = results.as_numpy(self._output_name)
            all_scores.extend(scores.tolist())

        return all_scores

    def compress_documents(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        """Rerank documents based on the query."""
        text_pairs = [(query, doc.page_content) for doc in documents]
        scores = self.score(text_pairs)
        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc, score in sorted_docs[:top_n]:
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = score
            reranked_docs.append(doc_copy)

        return reranked_docs


if __name__ == "__main__":
    # Example usage
    reranker = TritonCrossEncoder(
        triton_url="localhost:8001", model_name="bge-reranker-v2-m3"
    )
    query = "What is the capital of France?"
    documents = [
        Document(page_content="Berlin is the capital of Germany."),
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Madrid is the capital of Spain."),
    ]
    tic = time.time()
    reranked_docs = reranker.compress_documents(query, documents)
    for doc in reranked_docs:
        print(doc.page_content, doc.metadata.get("relevance_score"))
    toc = time.time()
    inference_time = round(toc - tic, 2)
    print(f"Inference time: {inference_time} seconds")
