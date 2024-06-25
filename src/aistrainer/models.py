from transformers import AutoConfig, AutoTokenizer


class BaseModel:
    def __init__(self, base_model_id, config):
        self.base_model_id = base_model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False)

    def apply_chat_template(self, record):
        pass


class Phi3Model(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.target_modules = ["o_proj", "qkv_proj", "gate_up_proj", "down_proj", "lm_head"]

    def apply_chat_template(self, record):
        inst = record["instruct"]
        inpt = record["input"]
        outp = record["output"]
        return f"""<s><|system|>
{inst}<|end|>
<|user|>{inpt}<|end|>
<|assistant|>{outp}<|end|>
"""


class CommandRModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

    def apply_chat_template(self, record):
        chat = [
            {"role": "system", "content": record["instruct"]},
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class Qwen2Model(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def apply_chat_template(self, record):
        chat = [
            {"role": "system", "content": record["instruct"]},
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class ModelsFactory:
    def __init__(self):
        pass

    def get_model_config(self, base_model_id):
        config = AutoConfig.from_pretrained(base_model_id)

        if config.model_type == "phi3":
            return Phi3Model(base_model_id, config)
        elif config.model_type == "cohere":
            return CommandRModel(base_model_id, config)
        elif config.model_type == "qwen2":
            return Qwen2Model(base_model_id, config)
        else:
            raise Exception("Unsupported model type: " + base_model_id)
