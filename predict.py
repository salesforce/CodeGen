"""A python module to generate code completions using CodeGen from salesforce."""
import os
import random
import tempfile
from typing import List, Optional

import cog
from cog import BaseModel, BasePredictor, Input, Path
from torch import device, float16, inference_mode

from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM
from jaxformer.hf.sample import (create_custom_gpt2_tokenizer, print_time,
                                 sample, set_seed, truncate)

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # huggingfaces transformers lib will cause problems without setting this env var

DEVICE = device("cuda")
CHECKPOINT_PATH = cog.Path("codegen-6B-mono")
PREDICTION_PATH = cog.Path(tempfile.gettempdir()).joinpath("prediction.markdown")
PAD_TOKEN = 50256
NUMPY_CONTEXT = """# import libraries

import numpy as np

"""
DEFAULT_CONTEXT = """# Implement a function that computes the square of an integer argument.\n"""


class CodeGenOutput(BaseModel):
    """Helper Output class for CodeGen. Allows for output to be markdown file or string."""
    markdown: Optional[Path]
    raw_output: Optional[str]


def save_to_markdown(
    completion_batch: List,
    context: str
) -> str:
    """Save a list of completion strings as markdown code snippets."""
    code_snippets = []
    for batch_idx, completion in enumerate(completion_batch):
        sequence_comment = ""
        if len(completion_batch) > 1:
            sequence_comment = (
                "# sequence " + f"{batch_idx + 1}/{len(completion_batch)}" + "\n"
            )
        truncation = truncate(completion).strip()
        if len(context) > 0:
            truncation = context + truncation
        code_snippet = "```py\n" + sequence_comment + truncation + "\n```"
        code_snippets.append(code_snippet)

    code_snippets = "\n".join(code_snippets)  # separate each code block by a newline

    with open(PREDICTION_PATH, encoding="utf-8", mode="w") as markdown_file:
        markdown_file.write(code_snippets)
    return code_snippets


def completions_as_raw_output(completion_batch: List, context: str) -> str:
    """Join completions with `======` separator. No markdown formatting."""
    code_snippets = []
    for completion in completion_batch:
        truncation = truncate(completion)
        if len(context) > 0:
            truncation = context + truncation
        code_snippets.append(truncation)
    return "======\n\n" + "\n\n======\n\n".join(code_snippets) + "\n\n======"


class Predictor(BasePredictor):
    """
    Predictor for `codegen-6B-mono` model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None

    def setup(self):
        """
        Initialize model weights, tokenizer, etc. in the setup method.
        """
        with print_time("loading parameters to CPU"):
            self.model = CodeGenForCausalLM.from_pretrained(
                CHECKPOINT_PATH,
                revision="float16",
                torch_dtype=float16,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

        with print_time("loading parameters to GPU"):
            self.model.to(DEVICE)

        with print_time("loading tokenizer"):
            self.tokenizer = create_custom_gpt2_tokenizer()
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = PAD_TOKEN

    @inference_mode()
    def predict(
        self,
        context: str = Input(
            description="Some starting python code. CodeGen will try to complete the code provided. Providing examples of what you want to do before your prompt can improve performance.",
            default=DEFAULT_CONTEXT,
        ),
        prepend_context_to_output: bool = Input(
            description="Whether to prepend your input to the output.",
            default=True,
        ),
        num_return_sequences: int = Input(
            description="Number of code completions to generate from context.",
            ge=1,
            le=10,
            default=1,
        ),
        temperature: float = Input(
            description="Increase to improve diversity of outputs, may cause artifacts.",
            ge=0,
            le=1,
            default=0.2,
        ),
        prepend_imports: bool = Input(
            description="Whether to prepend a numpy import to the context as in the paper.",
            default=True,
        ),
        top_p: float = Input(
            description="Top-p sampling probability.", ge=0, le=1, default=0.95
        ),
        max_length: int = Input(
            description="Max length of returned sequences.",
            ge=32,
            le=2048,
            default=128,
        ),
        seed: int = Input(
            description="Seed for reproducibility. Use -1 for a random seed.",
            default=-1,
        ),
        raw_output: bool = Input(
            description="Whether to return a long string (multiple code snippets separated by the string `======`) or a markdown url to be downloaded. May find useful for api.",
            default=False,
        ),
    ) -> CodeGenOutput:
        """Predict a code snippet given some starting context."""

        # Use a random seed by default
        if seed != -1:
            seed = int(seed)
        else:
            seed = random.randint(0, 2147483647)
        set_seed(seed, deterministic=True)
        print(f"Set seed {seed}")

        if prepend_imports:
            print("Prepending numpy import snippet to context")
            context = NUMPY_CONTEXT + context

        # If the last line of the context is a "comment" (i.e. starts with "#"), add a newline to the context.
        if context.split("\n")[-1].startswith("#"):
            print("Adding newline to context because last line is a comment")
            context += "\n"

        with print_time("Generating code completion"):
            completion_batch = sample(
                device=DEVICE,
                model=self.model,
                tokenizer=self.tokenizer,
                context=context,
                pad_token_id=PAD_TOKEN,
                num_return_sequences=num_return_sequences,
                temp=temperature,
                top_p=top_p,
                max_length_sample=max_length,
            )
            input_code = context if prepend_context_to_output else ""
            if raw_output:
                code_snippets = completions_as_raw_output(completion_batch, input_code)
                print(code_snippets)
                return CodeGenOutput(raw_output=code_snippets)
            else:
                code_snippets = save_to_markdown(completion_batch, input_code)
                print("=====\n\n")
                print(code_snippets)
                print("\n\n=====")
                return CodeGenOutput(markdown=PREDICTION_PATH)
