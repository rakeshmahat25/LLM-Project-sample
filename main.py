from transformers import AutoModelForCausalLM, AutoTokenizer

import os
os.environ['HF_HOME'] = 'E:/huggingface'

model_name = "meta-llama/Llama-2-7b-hf"  

model = AutoModelForCausalLM.from_pretrained(model_name, token="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")
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = generate_response("Hello! How can I assist you today?")
print(response)


from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
train_dataset = load_dataset("daily_dialog", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()