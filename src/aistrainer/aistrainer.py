import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import torch
import numpy as np
import torch._dynamo
import logging
import json
import gc
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from aistrainer.models import ModelsFactory


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    deepspeed.init_distributed()


class Aist:
    def __init__(self, base_model_id):
        self.base_model_id = base_model_id
        self.model_config = ModelsFactory().get_model_config(base_model_id)
        self.tokenizer = self.model_config.tokenizer
        self.device_map = "auto"
        self.deepspeed = None
        if torch.cuda.is_available():
            self.device_map = None
            self.deepspeed = self.get_deepspeed_config()
        self.bf16 = False
        self.fp16 = False
        self.attn_implementation = None
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                self.bf16 = True
                self.attn_implementation = "flash_attention_2"
            else:
                self.fp16 = True
        else:
            self.bf16 = True

    def get_instruction(self, record):
        return self.model_config.apply_chat_template(record)

    def map_func(self, record):
        return self.tokenizer(self.get_instruction(record),
                              truncation=True,
                              max_length=self.max_len,
                              padding="max_length")

    def filter_func(self, record):
        return len(self.tokenizer(self.get_instruction(record))["input_ids"]) < self.max_len

    def prepare_dataset(self, dataset, eval=False, max_len_percentile=95):
        self.eval = eval
        dataset = datasets.load_dataset(dataset, split="train")
        inputs = []

        for record in dataset:
            inputs.append(self.tokenizer(self.get_instruction(record))["input_ids"])

        target_lenghts = [len(x) for x in inputs]
        self.max_len = int(np.percentile(target_lenghts, max_len_percentile))

        logger.info(f"Dataset max_len detected: {self.max_len}")
        logger.info("Dataset example row after appy chat template:")
        logger.info("---------------------------------------------")
        logger.info(self.get_instruction(dataset[0]))
        logger.info("---------------------------------------------")
        dataset = dataset.filter(self.filter_func)
        dataset = dataset.map(self.map_func)
        logger.info(dataset["input_ids"][0])
        logger.info("---------------------------------------------")
        dataset = dataset.remove_columns(["instruct", "input", "output"])

        if eval:
            dataset = dataset.train_test_split(test_size=0.1)
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["test"]
            logger.info("Eval dataset:")
            logger.info(self.eval_dataset)
        else:
            self.train_dataset = dataset

        logger.info("Train dataset:")
        logger.info(self.train_dataset)

    def train(self, adapter_name, rank=32, lora_alpha=64, num_train_epochs=1, batch_size=4, gradient_steps=2):
        if not self.train_dataset:
            raise Exception("Dataset is not initialized")
        evaluation_strategy = "no"
        eval_steps = None
        eval_dataset = None

        if self.eval:
            evaluation_strategy = "steps"
            eval_steps = 0.1
            eval_dataset = self.eval_dataset

        logger.info("===================================================")
        logger.info("AISTrainer hyperparameters:")
        logger.info(f"fp16: {self.fp16}")
        logger.info(f"bf16: {self.bf16}")
        logger.info(f"flash_attentions: {self.attn_implementation}")
        logger.info(f"batch_size: {batch_size}")
        logger.info(f"gradient_steps: {gradient_steps}")
        if self.deepspeed is None:
            logger.info("deepspeed: disabled")
        else:
            logger.info("deepspeed: enabled")
        logger.info(f"lora_rank: {rank}")
        logger.info(f"lora_alfa: {lora_alpha}")
        logger.info("dora: disabled")
        logger.info(f"target_modules: {self.model_config.target_modules}")
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        logger.info(f"visible_devices: {visible_devices}")
        logger.info("---------------------------------------------------")
        logger.info("CUDA Devices:")
        for i in range(0, torch.cuda.device_count()):
            logger.info("GPU: " + str(i))
            logger.info("Total GPU Memory: " + str(torch.cuda.get_device_properties(i).total_memory))
            logger.info("Reserved GPU Memory: " + str(torch.cuda.memory_reserved(i)))
            logger.info("Allocated GPU Memory: " + str(torch.cuda.memory_allocated(i)))
            logger.info("---")
        logger.info("===================================================")

        args = TrainingArguments(".",
                                 num_train_epochs=num_train_epochs,
                                 logging_steps=1,
                                 eval_strategy=evaluation_strategy,
                                 eval_steps=eval_steps,
                                 gradient_checkpointing=True,
                                 bf16=self.bf16,
                                 fp16=self.fp16,
                                 per_device_train_batch_size=batch_size,
                                 per_device_eval_batch_size=batch_size,
                                 gradient_accumulation_steps=gradient_steps,
                                 eval_accumulation_steps=1,
                                 deepspeed=self.deepspeed)

        lora_config = LoraConfig(r=rank,
                                 lora_alpha=lora_alpha,
                                 target_modules=self.model_config.target_modules,
                                 lora_dropout=0.05,
                                 # use_rslora=rs_lora,
                                 # use_dora=True,
                                 # init_lora_weights="pissa",
                                 task_type="CAUSAL_LM")

        base_model = self.load_base_model()
        peft_model = get_peft_model(base_model, lora_config)
        trainer = Trainer(model=peft_model,
                          train_dataset=self.train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
                          args=args)
        trainer.train()
        trainer.save_model(adapter_name)

        del peft_model
        del base_model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        get_accelerator().empty_cache()
        deepspeed.runtime.utils.empty_cache()

    def merge(self, merged_name, *adapters):
        first_adapter = adapters[0]
        base_model = self.load_base_model(False)
        peft_model = PeftModel.from_pretrained(base_model, first_adapter, torch_dtype=torch.bfloat16)

        if len(adapters) > 1:
            names = ["default"]
            weights = [1.0]

            for i in range(1, len(adapters)):
                peft_model.load_adapter(adapters[i], adapter_name=str(i))
                names.append(str(i))
                weights.append(1.0)

            logger.info(f"Merging adapters: {names} to {merged_name}")

            peft_model.add_weighted_adapter(adapters=names,
                                            weights=weights,
                                            adapter_name="merged",
                                            combination_type="cat")

            merged_model = peft_model.merge_and_unload(adapter_names=["merged"])
        else:
            logger.info(f"Merging adapter {first_adapter} to {merged_name}")
            merged_model = peft_model.merge_and_unload()

        merged_model.save_pretrained(merged_name)

        # get original tokenizer for save
        tmp_tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tmp_tokenizer.save_pretrained(merged_name)
        try:
            tmp_tokenizer.save_vocabulary(merged_name)
        except:
            pass

        del merged_model
        del peft_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    def load_base_model(self, gradient_checkpointing=True):
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id,
                                                               use_cache=False,
                                                               torch_dtype=torch.bfloat16,
                                                               attn_implementation=self.attn_implementation,
                                                               device_map=self.device_map)
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_disable()
        return self.base_model

    def get_deepspeed_config(self):
        json_str = '''
        {
            "zero_force_ds_cpu_optimizer": false,
            "bf16": {"enabled": "auto"},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
                "overlap_comm": true,
                "reduce_bucket_size": "auto",
                "contiguous_gradients": true,
                "sub_group_size": 1e9,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_gather_16bit_weights_on_model_save": true
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto"
        }
        '''
        return json.loads(json_str)
