# CodeGen2

Official research release for the **CodeGen2** models (`1B`, `3B`, `7B`, `16B`) for **Program Synthesis** as presented in ICLR 2023:

*Title*: [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309)

*Authors*: [Erik Nijkamp](https://enijkamp.github.io/)\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en), [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ&hl=en), and [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en) (* indicates equal contribution)

## Hugging Face Integration

Model checkpoints are published at Hugging Face Hub.

* [CodeGen2-1B](https://huggingface.co/Salesforce/codegen2-1B)
* [CodeGen2-3.7B](https://huggingface.co/Salesforce/codegen2-3_7B)
* [CodeGen2-7B](https://huggingface.co/Salesforce/codegen2-7B)
* [CodeGen2-16B](https://huggingface.co/Salesforce/codegen2-16B)

Model cards outline how to use the model for causal and infill sampling.

## Sampling

Program synthesis in the form of auto-regressive sampling can be performed as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-7B")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-7B", trust_remote_code=True, revision="main")
inputs = tokenizer("# this function prints hello world", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
```

## Citation

```bibtex
@article{Nijkamp2023codegen2,
  title={CodeGen2: Lessons for Training LLMs on Programming and Natural Languages},
  author={Nijkamp, Erik and Hayashi, Hiroaki and Xiong, Caiming and Savarese, Silvio and Zhou, Yingbo},
  journal={arXiv preprint},
  year={2023}
}
```
