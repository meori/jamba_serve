from typing import Union
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/ml_workspace/Jamba-v0.1",
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("/ml_workspace/Jamba-v0.1", config = "/ml_workspace/Jamba-v0.1/tokenizer_config.json")

app = FastAPI()

class Request(BaseModel):
    content: str
    max_tokens: int

    
def predict_text(input_text, max_tokens=300):
    if input_text is None:
        raise Exception("content is empty")
    input_ids = tokenizer(input_text, return_tensors='pt').to(model.device)["input_ids"]
    outputs = model.generate(input_ids, max_new_tokens=max_tokens)

    predicted_text = tokenizer.batch_decode(outputs)

    return predicted_text

    

@app.get("/")
def redirect():
    return RedirectResponse(url='/docs')
 

@app.post("/completions")
def complete(request: Request):
    try:
        input_text = request.content
        if input_text is None:
            raise Exception("content is empty")
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
        
        predicted_text = greedy_decoding(input_ids, request.max_tokens)
        return {"content": predicted_text}
    except Exception as e:
        return {"error": str(e)}
        


