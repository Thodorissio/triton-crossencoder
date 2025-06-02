import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class TritonPythonModel:
    """Triton Python backend for BGE cross-encoder models."""

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = model_config = json.loads(args["model_config"])

        self.model_path = os.path.join(args["model_repository"], args["model_version"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)
        self.model.eval()

        self.activation = torch.nn.Sigmoid()

        # Get input/output names from config
        input_config = pb_utils.get_input_config_by_name(model_config, "INPUT_PAIRS")
        output_config = pb_utils.get_output_config_by_name(model_config, "SCORES")

        # Store data types
        self.input_dtype = pb_utils.triton_string_to_numpy(input_config["data_type"])
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Execute inference on a batch of requests."""
        responses = []

        for request in requests:
            input_data = pb_utils.get_input_tensor_by_name(
                request, "INPUT_PAIRS"
            ).as_numpy()

            text_pairs = []
            for pair in range(input_data.shape[0]):
                text_pairs.append(
                    [
                        input_data[pair][0].decode("utf-8"),
                        input_data[pair][1].decode("utf-8"),
                    ]
                )

            inputs = self.tokenizer(
                text_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                print(f"Model outputs: {outputs}")
                logits = self.activation(outputs.logits)
                scores = logits.cpu().numpy().flatten()

            output_tensor = pb_utils.Tensor("SCORES", scores.astype(self.output_dtype))

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        print("Cleaning up cross-encoder model...")
