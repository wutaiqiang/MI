Official code for ACL2026 Oral paper `Revisiting Model Interpolation for Efficient Reasoning`

<!-- <a href="https://huggingface.co/collections/taki555/timber-68db45e4f4c64c1bfe52b492"><b>[🤗 HF Models]</b></a> • -->
<a href="https://arxiv.org/abs/2510.10977"><b>[📜 Paper]</b></a> • 
<a href="https://github.com/wutaiqiang/MI"><b>[🐱 GitHub]</b></a>

# Environment

Please follow the official guidance of [Opencompass](https://github.com/open-compass/opencompass?tab=readme-ov-file#-environment-setup) to set up a Python environment.

We use the lmdeploy backend. Please remember to set
```
pip install "opencompass[lmdeploy]"
```

# Weights
Download the official weights from HuggingFace:

- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)

We recommend downloading via the huggingface-cli, such as

```
hf download Qwen/Qwen3-30B-A3B-Thinking-2507 --token $your_hf_token --local-dir weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Thinking-2507

hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --token $your_hf_token --local-dir weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507
```

Then, run the mi.py:

```
python mi.py --model_b /path/to/your/projects/base_model --model_i /path/to/your/projects/finetuned_model --lambda_val 0.5 --output_dir /path/to/your/projects/merged_output
```

where `lambda_val` is the interpolation factor.

# Evaluation

We employ the OpenCompass for evaluation.

You need to modify the config files first.


For example,  in `evaluation/qwen3_AIME.py`, replace the `paths` with your folder, modify the `gpus` to fit your machine.

Then all you need is to run `opencompass evaluation/qwen3_AIME.py` and wait the final results.

> Warnning: In this repo, we benchmark the Instruct-2507/Thinking-2507 version, which do not require setting 'enable_thinking'.
> If you try to evaluate the Qwen3 hybrid thinking model, such as Qwen3-4B, please fix the bugs in Opencompass and pass an extra enable_thinking following [this repo](https://github.com/wutaiqiang/Timber). 

# License

We use the Apache‑2.0 license.  Please also comply with the licenses of any upstream models and datasets.

# ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@inproceedings{wu2026revisiting,
  title={Revisiting model interpolation for efficient reasoning},
  author={Wu, Taiqiang and Yang, Runming and Liu, Tao and Wang, Jiahao and Wong, Ngai},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={8624--8638},
  year={2026}
}
```

For any questions, please pull an issue or email at `takiwu@connect.hku.hk`
