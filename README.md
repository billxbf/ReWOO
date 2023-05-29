# ReWOO
Official implementation for paper: _ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models._
This is a tool-augmented LM paradigm, leveraging _foreseeable reasoning_ ability of language models to improve system parameter and prompt efficiency.


## Installation
```
pip install -r requirements.txt
```
Generate API keys from [OpenAI](https://openai.com/blog/openai-api), [WolframAlpha](https://products.wolframalpha.com/api) and [SerpApi](https://serpapi.com/). Then save the keys to `./keys/openai.key`, `./keys/wolfram.key` and `./keys/serpapi.key` respectively.


## Single Run
```
 python run.py --method rewoo --toolset Google LLM --base_lm text-davinci-003 
```
Use `--method` to choose your prompt paradigm among `'direct', 'cot', 'react', 'rewoo'`

Use `--toolset` to provide available tools, including `'Google', 'Wikipedia', 'WolframAlpha', 'LLM', 'Calculator', 'SearchSOTU'`

Use `--base_lm` to choose a base language model, can be either `gpt-3.5-turbo`, `text-davinci-003` or `directory_to_alpca-lora_adapter`. You can also individually assign `--planner_lm` and `--solver_lm` for `rewoo`. 

Add `--print_trajectory` to print intermediate reasoning.


## Batch Evaluation on Benchmarks
```
python run_eval.py --method rewoo --dataset hotpot_qa --sample_size 10 --toolset Wikipedia LLM --base_lm gpt-3.5-turbo --save_result`
```

Use `--sample_size` to specify number of samples to evaluate.

Use `--save_result` to save evaluation results to `./results/`.

## Data & Model

Except public NLP benchmarks, we also evaluate ReWOO on a curated QA dataset over SOTU2023 [here](https://huggingface.co/datasets/rewoo/sotu_qa_2023). It is especially useful for ALMs to examine the model's tool-augmented ability in answering over private document.

We uploaded our instruction-tuned Planner 7B model (based on Alpaca 7B) [here](https://huggingface.co/rewoo/planner_7B). The instruction dataset (available [here](https://huggingface.co/datasets/rewoo/planner_instruction_tuning_2k/blob/main/README.md)) is a mixture of "correct" HotpotQA and TriviaQA task planning trajectories in ReWOO Framework.
