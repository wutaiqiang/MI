from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask



with read_base():

    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets

# repeat 4 times
for k,v in list(locals().items()):
    if k.endswith('_datasets'):
        # print("setting",k,v)
        v[0]["n"] = 8
    
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

work_dir = f'logs/Qwen3-30B-64k'

from opencompass.models import TurboMindModelwithChatTemplate, TurboMindModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

# input .sh output "##### Evaluation list #####" below
paths = [
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507"
]
paths_think = [
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Thinking-2507",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-02",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-04",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-045",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-05",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-055",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-06",
    "weights/Qwen3-30B-A3B/Qwen3-30B-A3B-Ins-Thi-08",
]

Instruct_settings = []
for p in paths:
    n = p.split("/")[-1]
    Instruct_settings.append((n,p))

Thinking_settings = []
for p in paths_think:
    n = p.split("/")[-1]
    Thinking_settings.append((n,p))

print(Instruct_settings)
print(Thinking_settings)

models = []

ws=65536
gpus=8

for abbr, path in Instruct_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=ws+4096, max_batch_size=4096, tp=gpus),
            gen_config=dict(do_sample=True, temperature=0.7, top_k=20, top_p=0.8, max_new_tokens=ws, mini_p=0),
            max_seq_len=4096,
            max_out_len=ws,
            batch_size=2048,
            run_cfg=dict(num_gpus=gpus),
        )
    )   
for abbr, path in Thinking_settings:  ## classic 4096
    models.append(
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=abbr,
            path=path,
            engine_config=dict(session_len=ws+4096, max_batch_size=4096, tp=gpus),
            gen_config=dict(do_sample=True, temperature=0.6, top_k=20, top_p=0.95, max_new_tokens=ws, mini_p=0),
            max_seq_len=4096,
            max_out_len=ws,
            batch_size=2048,
            run_cfg=dict(num_gpus=gpus),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    )     

models = models