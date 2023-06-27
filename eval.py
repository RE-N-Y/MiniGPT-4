import argparse
import time
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from einops import rearrange, repeat, reduce, pack, unpack

import json
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from tqdm import tqdm


import evaluate
from minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class Engine:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        stop_words_ids = [torch.tensor([835]).to(self.model.device),
                          torch.tensor([2277, 29937]).to(self.model.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    @torch.no_grad()
    def inference(self, samples, accelerator):
        [codes] = samples['codes']
        _model = accelerator.unwrap_model(self.model)
        img_embeds, atts_img = _model.encode_img(samples['images'])
        vqa_prompt = '###Human: <Img><ImageHere></Img> '
        img_embeds, atts_img = _model.prompt_wrap(img_embeds, atts_img, vqa_prompt, add_special_tokens=True)
        PROMPT = "Execute the following code on the image. Log the output line by line and print the result.\n"

        tokens = _model.llama_tokenizer(PROMPT + codes, add_special_tokens=False, return_tensors='pt')
        token_embeds = _model.llama_model.model.embed_tokens(tokens['input_ids'].to(img_embeds.device))
        inputs_embeds = torch.cat([img_embeds, token_embeds], dim=1)
        outputs = _model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=1024,
            stopping_criteria=self.stopping_criteria,
            num_beams=1,
            do_sample=True,
            repetition_penalty=1.0, 
            length_penalty=1, 
            min_length=1,
            temperature=1.0,
            top_p=0.99,
        )
        
        outputs = _model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs

if __name__ == "__main__":
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    accelerator = Accelerator(mixed_precision="bf16")

    def collate_fn(batch):
        codes_outputs = [b['text_input'].split("OUTPUT\n") for b in batch]
        codes = [c[0] + "OUTPUT\n" for c in codes_outputs]
        outputs = [c[1] for c in codes_outputs]
        images = torch.stack([b['image'] for b in batch])
        image_ids = [b['image_id'] for b in batch]
        question_ids = [b['question_id'] for b in batch]

        return { 'codes': codes, 'outputs': outputs, 'images': images, 'image_id': image_ids, 'question_id': question_ids }

    dataloader = DataLoader(datasets["cc_sbu_align"]["train"], batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    engine = Engine(model, vis_processor)
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        outputs = engine.inference(batch, accelerator)
        [question_id] = batch['question_id']
        [references] = batch['outputs']
        [codes] = batch['codes']
        [predictions] = [o.replace("###", "") for o in outputs]

        rouge.add_batch(predictions=[predictions], references=[references])
        meteor.add_batch(predictions=[predictions], references=[references])

        with open(f"gqa/testdev/result/{question_id}.json", "w") as f:
            json.dump({
                "predictions": predictions, 
                "references": references, 
                "codes": batch["codes"],
            }, f)

    print("ROUGE", rouge.compute())
    print("METEOR", meteor.compute())




