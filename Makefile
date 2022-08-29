code_gen1:
	source /Users/bytedance/githubcode/CodeGen/.venv/bin/activate
	python3.8 -m jaxformer.hf.sample --model codegen-350M-multi --context "func HelloWorld"

code_gen2:
	source /Users/bytedance/githubcode/CodeGen/.venv/bin/activate
	python3.8 -m jaxformer.hf.sample --model codegen-350M-multi --context "func RecursiveVisitCategoryTree"