# Include Python
from python:3.11.1-buster

RUN python -m pip install --upgrade pip setuptools

WORKDIR /

ADD requirements.txt .

RUN pip install -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM;model_id = 'unikei/t5-base-split-and-rephrase';AutoTokenizer.from_pretrained(model_id);AutoModelForSeq2SeqLM.from_pretrained(model_id)"

COPY handler.py .

CMD [ "python", "-u", "/handler.py" ]

