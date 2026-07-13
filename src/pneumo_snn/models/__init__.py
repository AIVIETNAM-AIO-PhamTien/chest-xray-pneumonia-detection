# Model blocks (Table 1 / Fig.2-6 of the paper). Suggested file-per-block layout:
#   spatial_cnn.py      -> Spatial Feature Extraction block (Fig.2)
#   temporal_gru.py      -> Temporal Dynamics Modeling / Bi-GRU block (Fig.3)
#   spiking.py            -> Spiking Neural Processing / LIF block (Fig.4)
#   decision_head.py    -> Decision Head block (Fig.5)
#   attention.py          -> Attention map module
#   hybrid_model.py      -> Full pipeline assembly (Fig.6)
# See docs/paper_mapping.md for ownership/naming conventions.
