from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="devJy/SEEDRA-zero-small",
    torch_dtype=torch.bfloat16
)

sentence = "I have to go to work, but I'm so sleepy that I want to sleep more instead of going to work."

SYSTEM_PROMPT = """You are a similar sentence generation  Assistant.
For the instructions below, output **only** an array of similar sentences in JSON format..
Example output: ["similar_sentence1","similar_sentence2"]"""

messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sentence},
]

output = pipe(messages, max_new_tokens=4096)

print(output[0]["generated_text"][-1]["content"])

# terminal output 
# ["I need to go to work, but I'm feeling so tired that I'd rather sleep than go to work.",
#  "I have to go to work, but I'm so drowsy that I'd rather sleep than go to work.",
#  "I have to go to work, but I'm so fatigued that I'd rather sleep than go to work.",
#  "I need to go to work, but I'm so exhausted that I'd rather sleep than go to work.", 
#  "I have to go to work, but I'm so sleepy that I'd rather sleep than go to work."]