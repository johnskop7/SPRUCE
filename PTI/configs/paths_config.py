"""Paths used by the PTI inversion pipeline.

Replace the placeholders below with paths appropriate for your environment.
"""

# Pretrained models
e4e = "/path/to/e4e_checkpoint.pt"
stylegan2_ada = "/path/to/stylegan2_ada.pkl"

# Output directories
checkpoints_dir = "/path/to/pti_generators"
embedding_base_dir = "/path/to/embeddings"

# Input images
input_data_path = "/path/to/query_images"

# Experiment identifier used to organize inversion outputs
input_data_id = "experiment_name"

# Output keywords
pti_results_keyword = "PTI"
e4e_results_keyword = "e4e"
sg2_results_keyword = "SG2"
sg2_plus_results_keyword = "SG2_plus"
multi_id_model_type = "multi_id"
