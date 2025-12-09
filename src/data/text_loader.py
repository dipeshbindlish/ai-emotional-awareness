from datasets import load_dataset

def get_text_split():
    ds = load_dataset("emotion")
    return ds
