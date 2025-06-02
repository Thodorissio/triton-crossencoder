# Serve Crossencoder with Triton

In this example we will show how to deploy a Crossencoder model using Triton Inference Server and use it as a LangChain reranker. We will use the [BGE Reranker v2 model](https://huggingface.co/BAAI/bge-reranker-v2-m3), but this tutorial can be adapted to any Crossencoder model (with some possible changes in the model inference).

## Setup

To run this example you should copy the model files into the `model-repository/bge-reranker-v2-m3/1/` directory (they usually are in `.cache` dir). The model-repository structure should look like this:
```
model-repository/
└── bge-reranker-v2-m3/
    └── 1/
        ├── model.py
        ├── config.json
        ├── sentencepiece.bpe.model
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
    └── config.pbtxt
```

## Usage

1. Create and run the docker:
```bash
docker compose up
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the example script:
```bash
python reranker.py
```