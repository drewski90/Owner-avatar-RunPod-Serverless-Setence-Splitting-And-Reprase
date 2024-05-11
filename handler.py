from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import runpod
from json import dumps

# id of the huggingface model
model_id = "unikei/t5-base-split-and-rephrase"

# load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# sentence splitter instance
splitter = SentenceSplitter(language='en')

def encode_text(job):
    
    job_input = job["input"]
    prompt = job_input['prompt']
    
    complex_tokenized = tokenizer(
        prompt, 
        padding="max_length", 
        truncation=True,
        max_length=256, 
        return_tensors='pt'
    )
    
    simple_tokenized = model.generate(
        complex_tokenized['input_ids'], 
        attention_mask = complex_tokenized['attention_mask'], 
        max_length=256, 
        num_beams=5
    )
    simple_sentences = tokenizer.batch_decode(
        simple_tokenized, 
        skip_special_tokens=True
    )
    
    # split sentences from model ouput
    
    result = splitter.split(text=simple_sentences[0])
    
    return dumps({"result": result})


runpod.serverless.start({"handler": encode_text})