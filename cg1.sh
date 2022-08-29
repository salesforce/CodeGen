source .venv/bin/activate
python3.8 -m jaxformer.hf.sample --model codegen-350M-multi --context "$1"
