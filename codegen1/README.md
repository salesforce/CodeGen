# CodeGen1

Official research release for the **CodeGen1** models (`2B`, `6B`, `16B`) for **Program Synthesis** as presented in ICLR 2023:

*Title*: [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474)  

*Authors*: [Erik Nijkamp](https://enijkamp.github.io/)\*, [Bo Pang](https://scholar.google.com/citations?user=s9fNEVEAAAAJ&hl=en)\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Lifu Tu](https://home.ttic.edu/~lifu/), [Huan Wang](https://scholar.google.com/citations?user=7NpTttkAAAAJ&hl=en), [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en), [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ&hl=en), and [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en) (* indicates equal contribution)

## Hugging Face Integration

The models are available on the [HuggingFace Hub](https://huggingface.co/models?search=salesforce+codegen).

## Sampling

Program synthesis in the form of auto-regressive sampling can be performed as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
inputs = tokenizer("# this function prints hello world", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
```

## Citation

```bibtex
@article{nijkamp2022codegen,
  title={CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis},
  author={Nijkamp, Erik and Pang, Bo and Hayashi, Hiroaki and Tu, Lifu and Wang, Huan and Zhou, Yingbo and Savarese, Silvio and Xiong, Caiming},
  journal={ICLR},
  year={2023}
}
```
