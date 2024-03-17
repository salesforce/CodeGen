from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("checkpoints/codegen25-7b-multi", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("checkpoints/codegen25-7b-multi")
inputs = tokenizer("def hello_world():", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
