# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-3B")

if __name__ == "__main__":
    download_model()