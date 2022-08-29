source .venv/bin/activate
python3.8 -m jaxformer.hf.sample --model codegen-2B-multi --context "$1"
