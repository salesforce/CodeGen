from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("checkpoints/codegen2-6B")
model = CodeGenForCausalLM.from_pretrained("checkpoints/codegen2-6B", torch_dtype=torch.float16, revision="sharded")
inputs = tokenizer("# this function prints hello world", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
