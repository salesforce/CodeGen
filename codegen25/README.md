# CodeGen2.5

Official research release for the **CodeGen2.5** models for **Program Synthesis**.

Title: [**CodeGen2.5: Small, but mighty**](https://blog.salesforceairesearch.com/codegen25)

Authors: [Erik Nijkamp](https://eriknijkamp.com)\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en), [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en) (* equal contribution)

## Hugging Face Integration

Model checkpoints are published at Hugging Face Hub.

* [CodeGen2.5-7B-multi](https://huggingface.co/Salesforce/codegen25-7b-multi) (Apache-2.0)
* [CodeGen2.5-7B-mono](https://huggingface.co/Salesforce/codegen25-7B-mono) (Apache-2.0)
* [CodeGen2.5-7B-instruct](https://huggingface.co/Salesforce/codegen25-7B-instruct) (*Research purposes only*)

Model cards outline how to use the model for causal and infill sampling. Please refer to each model card for more details.

The models are pre-trained on the [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata), a programming language dataset developed by [BigCode](https://huggingface.co/bigcode).

## Requirements

```
transformers>=4.29.2
tiktoken==0.4.0
```

## Sampling

Program synthesis in the form of auto-regressive sampling can be performed as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-mono", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-mono")
inputs = tokenizer("def hello_world():", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```

## Citation

Please cite CodeGen2 paper:

```bibtex
@article{Nijkamp2023codegen2,
  title={CodeGen2: Lessons for Training LLMs on Programming and Natural Languages},
  author={Nijkamp, Erik and Hayashi, Hiroaki and Xiong, Caiming and Savarese, Silvio and Zhou, Yingbo},
  journal={ICLR},
  year={2023}
}
```
