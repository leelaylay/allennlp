{
  "dataset_reader": {
    "type": "mspars"
  },
  "validation_dataset_reader": {
    "type": "mspars"
  },
  "train_data_path": "dataset/MSParS/data/MSParS.train",
  "validation_data_path": "dataset/MSParS/data/MSParS.dev",
  "model": {
    "type": "simple_seq2seq",
    "max_decoding_steps": 5,
    "source_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 15
        }},
      "encoder": {
        "type": "lstm",
        "input_size": 15,
        "hidden_size": 128,
        "bidirectional": true
      }
    },
    "iterator": {
      "type": "bucket",
      "batch_size": 32,
      "sorting_keys": [["source_tokens", "num_tokens"]]
    },
  "trainer": {
    "num_epochs": 20,
    "patience": 5,
    "cuda_device": 0,
    "grad_norm": 5.0,
    "optimizer": {
      "type": "adam"
    }
  }
}
