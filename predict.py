import os
import tempfile

import torch
import cog
from cog import BasePredictor, Input
from torch import inference_mode
from torch.cuda.amp import autocast

from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM
from jaxformer.hf.sample import (create_custom_gpt2_tokenizer, print_time,
                                 sample, set_env, set_seed, truncate)

DEFAULT_CONTEXT = "# approximate pi using the monte carlo method \ndef calculate_pi():"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_name = "codegen-2B-mono"
        ckpt = f"./checkpoints/{self.model_name}"
        assert os.path.isdir(ckpt), f"Model directory {ckpt} does not exist"

        self.device = torch.device("cuda")

        with print_time("loading parameters"):
            self.model = CodeGenForCausalLM.from_pretrained(
                ckpt,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)

        with print_time("loading tokenizer"):
            self.tokenizer = create_custom_gpt2_tokenizer()
            self.tokenizer.padding_side = "left"
            self.pad = 50256
            self.tokenizer.pad_token = self.pad

    @autocast()
    @inference_mode()
    def predict(
        self,
        context: str = Input(
            description="Some starting python code. CodeGen will try to complete the code provided, up to the max_length.",
            default=DEFAULT_CONTEXT,
        ),
        rng_seed: int = Input(description="Random number generator seed", default=0),
        rng_deterministic: bool = Input(description="Deterministic RNG", default=True),
        top_p: float = Input(
            description="Top-p sampling probability.", ge=0, le=1, default=0.95
        ),
        temperature: float = Input(
            description="Temperature for sampling", ge=0, le=1, default=0.2
        ),
        max_length: int = Input(
            description="Maximum length of generated text", ge=0, le=1000, default=128
        ),
        batch_size: int = Input(description="Batch size", ge=1, le=10, default=1),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        set_env()
        set_seed(rng_seed, deterministic=rng_deterministic)

        with print_time("sampling"):
            completion = sample(
                device=self.device,
                model=self.model,
                tokenizer=self.tokenizer,
                context=context,
                pad_token_id=self.pad,
                num_return_sequences=batch_size,
                temp=temperature,
                top_p=top_p,
                max_length_sample=max_length,
            )[0]
            truncation = truncate(completion)

        # cog handles markdown files with the `.md` extension, 
        # so we need to write the output to a file inside after wrapping in a md code block.
        out_path = cog.Path(tempfile.mkdtemp()) / "codegen_prediction.md"
        output_as_markdown = f"```python\n{context}{truncation}\n```"
        with open(out_path, "w") as f:
            f.write(output_as_markdown)
        return out_path
