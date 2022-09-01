<p align="center">
  <img src="assets/codegen_logo.png" width="25%">
</p>

# CodeGen

Official release for the **CodeGen** models (`350M`, `2B`, `6B`, `16B`) for **Program Synthesis**. That is, the model **
translates English into executable code** as presented in the paper:

*Title*: [A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474)

*Authors*: [Erik Nijkamp](https://enijkamp.github.io/)\*
, [Bo Pang](https://scholar.google.com/citations?user=s9fNEVEAAAAJ&hl=en)\*, [Hiroaki Hayashi](https://hiroakih.me/)\*
, [Lifu Tu](https://home.ttic.edu/~lifu/), [Huan Wang](https://scholar.google.com/citations?user=7NpTttkAAAAJ&hl=en)
, [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en)
, [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ&hl=en),
and [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en) (* indicates equal contribution)

<p align="center">
  <img src="assets/two.gif" width="60%">
</p>

The current version releases the sampling code, while the detailed training code will be released soon.

## HuggingFace

The model is available on the [HuggingFace Hub](https://huggingface.co/models?search=salesforce+codegen) with a Colab
demo [here](https://colab.research.google.com/drive/11YU00W-JLNXn-3YckJGOSxFf_TQfCXYr?usp=sharing).

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
inputs = tokenizer("# this function prints hello world", return_tensors="pt").to(0)
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
```

## Colab

This
[Google Colab notebook](https://colab.research.google.com/drive/1fQI8OgzMAR0bquCrvhlAtXSw6iMFbVgI) allows for sampling
from the CodeGen models.

## Setup

```sh
pip3 install git+https://github.com/huggingface/transformers.git

git clone https://github.com/salesforce/CodeGen
cd CodeGen

# download the model parameters
# codegen-350M-nl,multi,mono
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-nl.tar.gz && tar -xvf checkpoints/codegen-350M-nl.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-multi.tar.gz && tar -xvf checkpoints/codegen-350M-multi.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/
# codegen-2B-nl,multi,mono
# wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-2B-nl.tar.gz && tar -xvf checkpoints/codegen-2B-nl.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-2B-multi.tar.gz && tar -xvf checkpoints/codegen-2B-multi.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-2B-mono.tar.gz && tar -xvf checkpoints/codegen-2B-mono.tar.gz -C checkpoints/
# codegen-6B-nl,multi,mono
# wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-6B-nl.tar.gz && tar -xvf checkpoints/codegen-6B-nl.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-6B-multi.tar.gz && tar -xvf checkpoints/codegen-6B-multi.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-6B-mono.tar.gz && tar -xvf checkpoints/codegen-6B-mono.tar.gz -C checkpoints/
# codegen-16B-nl,multi,mono
# wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-16B-nl.tar.gz && tar -xvf checkpoints/codegen-16B-nl.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-16B-multi.tar.gz && tar -xvf checkpoints/codegen-16B-multi.tar.gz -C checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-16B-mono.tar.gz && tar -xvf checkpoints/codegen-16B-mono.tar.gz -C checkpoints/

# create a virtual environment with requirements
python3.8 -m venv .venv

source .venv/bin/activate

pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample from the model with an arbitrary context
python3.8 -m aixcoder.hf.sample --model codegen-350M-mono --context "def hello_world():"

python3.8 -m aixcoder.hf.sample --model codegen-350M-mono --context "recursive visit a category tree"

python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func RecursiveVisitCategoryTree"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func KMP"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func ReverseSlice"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func InsertRedBlackTree"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func SearchSkipList"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func MergeBinaryTree"
python3.8 -m aixcoder.hf.sample --model codegen-350M-multi --context "func BatchGetRecordsByIdList"

python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func RecursiveVisitCategoryTree"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func KMP"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func ReverseSlice"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func MergeBinaryTree"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func SearchSkipList"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func SortMapByValue"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func SortSlice"
python3.8 -m aixcoder.hf.sample --model codegen-2B-multi --context "func BatchGetRecordsByIdList"

# 内存扛不住
python3.8 -m aixcoder.hf.sample --model codegen-6B-multi --context "func HelloWord"
python3.8 -m aixcoder.hf.sample --model codegen-6B-multi --context "func InsertRedBlackTree"
python3.8 -m aixcoder.hf.sample --model codegen-6B-multi --context "func RecursiveVisitCategoryTree"
python3.8 -m aixcoder.hf.sample --model codegen-6B-multi --context "func MergeBinaryTree"

```

# code generated

```go 

func HelloWorld() string {
        return "Hello World!"
}

func main() {
        hello := HelloWorld()
        fmt.Println(hello)
}


func RecursiveVisitCategoryTree(root *CategoriesTreeNode, callback func(CategoryTreeNode)) {
        for _, category := range root.Children {
                callback(category)
                RecursiveVisitCategoryTree(category, callback)
        }
}

// 350M - 115s
func ReverseSlice(a []int) []int {
        b := make([]int, len(a))
        copy(b, a)
        for i := len(a) - 1; i >= 0; i-- {
                b[i], b[i+1] = b[i+1], b[i]
        }
        return b
}

// 2B - 354s
func ReverseSlice(s []int) []int {
        for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
                s[i], s[j] = s[j], s[i]
        }
        return s
}


// 2B - 355s
func MergeBinaryTree(root *TreeNode, left *TreeNode, right *TreeNode) *TreeNode {
        if root == nil {
                return root
        }
        if left == nil {
                return root
        }
        if right == nil {
                return root
        }
        if left.Val < right.Val {
                root.Val = left.Val
                root.Left = MergeBinaryTree(root.Left, left, right)
        } else {
                root.Val = right.Val
                root.Right = MergeBinaryTree(root.Right, left, right)
        }
        return root
}

func main() {
        root := &TreeNode{Val: 1}
        root.Left = &TreeNode{Val: 2}
        root.Right = &TreeNode{Val: 3}
        root.Left.Left = &TreeNode{Val: 4}
        root.Left.Right = &TreeNode{Val: 5}
        root.Right.Left = &TreeNode{Val: 6}
        root.Right.Right = &TreeNode{Val: 7}
        fmt.Println(root)
        fmt.Println(MergeBinaryTree(root, root.Left, root.Right))
}

====================================================================================================
sampling took 355.50s
====================================================================================================
func RecursiveVisitCategoryTree(root *CategoryTree, visitor Visitor) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTree(root.Left, visitor)
        visitor(root)
        RecursiveVisitCategoryTree(root.Right, visitor)
}

// RecursiveVisitCategoryTreeWithDepth visits the category tree rooted at root,
// calling visitor for each node with a depth parameter.
func RecursiveVisitCategoryTreeWithDepth(root *CategoryTree, visitor Visitor, depth int) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTreeWithDepth(root.Left, visitor, depth+1)
        visitor(root, depth)
        RecursiveVisitCategoryTreeWithDepth(root.Right, visitor, depth+1)
}

====================================================================================================
sampling took 388.86s
done.




func RecursiveVisitCategoryTree(root *CategoriesTreeNode, callback func(CategoryTreeNode)) {
        for _, category := range root.Children {
                callback(category)
                RecursiveVisitCategoryTree(category, callback)
        }
}

====================================================================================================
sampling took 110.80s
done.

```

## Released Models

We release models of various sizes trained on various datasets. The models are named in the following format:

```
codegen-{model-size}-{data}
```

`model-size` has 4 options: `350M`, `2B`, `6B`, `16B`, which represent the number of parameters in each model.

`data` has 3 options: `nl`, `multi`, `mono`.

* `nl` models are randomly initialized and trained on [The Pile](https://github.com/EleutherAI/the-pile), a 825.18 GB
  English text corpus.
* `multi` models are initialized from `nl` models and then trained on a corpus with code data consisting of multiple
  programming languages.
* `mono` models are initialized from `multi` models and then trained on a corpus with Python code data.

The model names can be provided to the `--model` flag for `sample.py`. See a sample usage above in Setup.

## Citation

If you find our code or paper useful, please cite the paper:

```bibtex
@article{Nijkamp2022ACP,
  title={A Conversational Paradigm for Program Synthesis},
  author={Nijkamp, Erik and Pang, Bo and Hayashi, Hiroaki and Tu, Lifu and Wang, Huan and Zhou, Yingbo and Savarese, Silvio and Xiong, Caiming},
  journal={arXiv preprint},
  year={2022}
}
```

## License

Our code is BSD-3 licensed. See LICENSE.txt for details.
