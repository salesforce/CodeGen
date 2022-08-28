code_gen1:
	python3.8 -m jaxformer.hf.sample --model codegen-350M-mono --context "def hello_world():"

code_gen2:
	python3.8 -m jaxformer.hf.sample --model codegen-350M-mono --context "recursive visit a category tree"