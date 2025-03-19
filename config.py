
def get_config():
    return {
        "seq_len": 1024,
        "vocab_size": 50257,
        "d_model": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout": 0.1,
        "qkv_bias":False
    }