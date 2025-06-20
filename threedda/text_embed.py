import torch
import numpy as np
import transformers

class CLIPTextEmbedder():

    def __init__(self, device="cpu"):
        self.device = device
        self.text_model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = self.text_model
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer.model_max_length = 53
        self.cache = {}
        self.to(device)

    def to(self, device):
        self.device = device
        self.text_model = self.text_model.to(self.device)
        return self
    
    def embed_instruction(self, instr):

        if instr in self.cache:
            return self.cache[instr]

        tokens = self.tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(self.device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            embed = self.text_model(tokens).last_hidden_state.cpu()

        self.cache[instr] = embed
        return embed