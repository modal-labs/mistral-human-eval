# Un Ministral, des grands singes de langage

![Des singes](assets/des_singes.jpg)

This repo demonstrates how to replicate the results of the [Large Language Monkeys paper](https://arxiv.org/abs/2407.21787)
using a different model, [Ministral 8B](https://mistral.ai/news/ministraux),
and a different dataset, [HumanEval](https://github.com/openai/human-eval).

It runs both the code generation model and the sandboxed code evaluation on [Modal](https://modal.com) and massively in parallel --
on Modal's free tier, that's code generation across 10 H100 GPUs running at an aggregate throughput of 5 - 10k tok/s per GPU
and code evaluation across over 100 Sandboxes.

## How-To

### Setup Modal

```bash
pip install modal  # that's it :)
modal setup  # if you're new to Modal
```

### Test and deploy inference on Modal

```bash
# test
modal run le_inference.py
# deploy
modal deploy le_inference.py
```

### Test and run the benchmark in parallel

```bash
# test
modal run le_client.py --dry-run --n 1 --subsample 1
# test and save results
modal run le_client.py --no-dry-run --n 1 --subsample 1
# run full dataset, 1000 attempts per problem
modal run le_client.py --no-dry-run --n 1000 --subsample 100 # percent
```

### Calculate results in parallel in sandboxes

```bash
# run concurrently or afterwards
modal run les_evals.py
```

### Analyze results

```bash
modal launch jupyter --volume mistral-humaneval --mount analysis
# run the notebook in `mount/`
```

## Other files

The `le_quant` and `le_quant_wrapper` scripts demonstrate language model quantization
with `llm-compressor` run on Modal.

We ran those already to generate the model used by default in the example,
so you don't need to run them, but they are included for completeness.
