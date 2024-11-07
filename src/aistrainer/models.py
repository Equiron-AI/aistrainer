import logging
from transformers import AutoConfig, AutoTokenizer


logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, base_model_id, config):
        self.base_model_id = base_model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False)

        # В документации по Cohere Aya указаны только модули "q_proj", "v_proj", "k_proj", "o_proj"
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        logger.info("Pad token: " + self.tokenizer.pad_token)
        logger.info("Bos token: " + self.tokenizer.bos_token)
        logger.info("Eos token: " + self.tokenizer.eos_token)

    def apply_chat_template(self, record):
        pass


class Gemma2Model(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        # self.response_template = "<start_of_turn>model"

    def apply_chat_template(self, record):
        if "text" in record and record["text"]:
            return record["text"]

        inst = record["instruct"]
        inpt = record["input"]
        outp = record["output"]
        return f"""<bos><start_of_turn>system
{inst}<end_of_turn>
<start_of_turn>user
{inpt}<end_of_turn>
<start_of_turn>model
{outp}<end_of_turn>
"""


class CohereModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        # self.response_template = "<|CHATBOT_TOKEN|>"

    def apply_chat_template(self, record):
        if "text" in record and record["text"]:
            return record["text"]

        inst = record["instruct"]
        inpt = record["input"]
        outp = record["output"]
        chat = [
            {"role": "system", "content": inst},
            {"role": "user", "content": inpt},
            {"role": "assistant", "content": outp}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class ModelsFactory:
    def __init__(self):
        pass

    def get_model_config(self, base_model_id):
        config = AutoConfig.from_pretrained(base_model_id)

        if config.model_type == "gemma2":
            return Gemma2Model(base_model_id, config)
        elif config.model_type == "cohere":
            return CohereModel(base_model_id, config)
        else:
            raise Exception("Unsupported model type: " + base_model_id)
