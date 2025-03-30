from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other models like "gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def ask_llm(content, question, max_length=150):
    # Combine content and question into a prompt
    prompt = f"{content}\n\nQuestion: {question}\nAnswer:"
    
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response using the model
    outputs = model.generate(
        inputs,
        max_length=max_length,  # Maximum length of the generated response
        num_return_sequences=1,  # Number of responses to generate
        no_repeat_ngram_size=2,  # Prevent repetition of n-grams
        temperature=0.7,         # Controls randomness (lower = more deterministic)
        top_k=50,                # Top-k sampling
        top_p=0.95,              # Nucleus sampling
        do_sample=True,          # Enable sampling
    )
    
    # Decode the generated response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part (remove the prompt)
    answer = answer.replace(prompt, "").strip()
    
    # Return the answer as a JSON response
    return json.dumps({"question": question, "answer": answer})

# Example usage
content = """
Docker is an open-source platform that allows developers to automate the deployment of applications inside lightweight, portable containers. It enables applications to run consistently across different environments, making it easier to develop, ship, and run applications. A container is a lightweight, standalone executable package that includes everything needed to run an application (code, runtime, system tools, libraries, dependencies).
Containers ensure that the application runs the same way in different environments. A Docker Image is a read-only template used to create containers.
"""

question = "provie me content topic?"
response = ask_llm(content, question)

print(response)
