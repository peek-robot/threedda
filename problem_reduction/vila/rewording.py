import re
import torch
import json
import transformers

from huggingface_hub import login
login(token = "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")

class RewordLLM():
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", n_samples=1, top_p=0.9, temperature=0.7, num_return_sequences=1):

        self.n_samples = n_samples
        self.top_p = top_p
        self.temperature = temperature
        self.num_return_sequences = num_return_sequences

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        self.eos_token_id = self.pipeline.tokenizer.encode('<STOP>')[2]

        self.placeholder = "XXX"
        # self.prompt = f'Provide {self.n_samples} alternative formulations for the following instruction maintaining its meaning (object, location, left/right, semantics) and length. Occasionally, replace the object names with synonyms, e.g, "apple" -> "healthy fruit", "cup" -> "ceramic mug", "Taylor Swift" -> "most successful female artist". Respond with a single json document, e.g., {{"responses": ["response 1", ...]}}. End your response with <STOP>. INSTRUCTION: {self.placeholder} RESPONSE: {{"responses": ["'
        # self.prompt = f'Provide {self.n_samples} alternative formulations for the following instruction maintaining its meaning (object, location, left/right, semantics). Replace object names with synonyms and descriptive adjectives. Respond with a single json document, e.g., {{"responses": ["response 1", ...]}}. End your response with <STOP>. INSTRUCTION: {self.placeholder} RESPONSE: {{"responses": ["'
        self.prompt = f'Provide {self.n_samples} alternative formulations for the following instruction maintaining its meaning (object, spatial location, semantics). Replace object names with synonyms and descriptive adjectives if applicable. Respond with a single json document, e.g., {{"responses": ["response 1", ...]}}. End your response with <STOP>. INSTRUCTION: {self.placeholder} RESPONSE: {{"responses": ["'
        

    def extract_json(self, result):
        
        # extract json
        pattern = r"RESPONSE: \{.*?\}"
        matches = re.findall(pattern, result[0]["generated_text"])
        result_list = json.loads(matches[0].split("RESPONSE: ")[-1])["responses"]

        return result_list

    def reword(self, text):
        
        # add instruction to prompt
        prompt = self.prompt.replace(self.placeholder, text)

        # retry until valid answer is generated
        while True:
            # generate response
            result = self.pipeline(
                        prompt,
                        num_return_sequences=self.num_return_sequences,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        eos_token_id=self.eos_token_id,
                        pad_token_id=self.eos_token_id,
                        max_new_tokens=256,
                )
            try:
                # extract json
                return self.extract_json(result)
            except Exception as e:
                print(e)
                continue