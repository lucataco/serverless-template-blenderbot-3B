import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-3B").to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    reply_ids = model.generate(**inputs)

    result = tokenizer.batch_decode(reply_ids)

    # Return the results as a dictionary
    return result
