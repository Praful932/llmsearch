import torch, os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Union, Dict
from transformers import AutoConfig, PretrainedConfig
from transformers.generation.utils import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Lora
#https://github.com/turboderp/exllamav2/issues/232

class ExLlamaV2ForCausalLM(GenerationMixin):

    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        exllama_config: ExLlamaV2Config,
        model: ExLlamaV2,
        loras: Dict[str, ExLlamaV2Lora] = {'': None},
        active_adapter: str = '',
        **kwargs
    ):
        self.config = config
        self.generation_config = generation_config
        self.exllama_config = exllama_config
        self.model = model
        self.loras = loras
        if '' not in self.loras:
            self.loras[''] = None
        self._active_adapter = active_adapter
        self._adapter_enabled = True
        if active_adapter == '':
            self.disable_adapter_layers()

    def can_generate(self):
        return True

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    @property
    def main_input_name(self) -> str:
        return 'input_ids'

    @property
    def active_adapters(self) -> List[str]:
        return [self._active_adapter] if self._adapter_enabled else []

    @property
    def active_adapter(self) -> List[str]:
        return self._active_adapter if self._adapter_enabled else ''

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[ExLlamaV2Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_size: int = -1,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loras = self.loras.get(self.active_adapter, None)
        loras = [loras] if loras else loras

        if labels is None:
            if past_key_values is None:
                past_key_values = ExLlamaV2Cache(self.model, input_ids.shape[0], cache_size)
                self.model.forward(input_ids[...,:-1], past_key_values, preprocess_only=True, loras=loras, input_mask=attention_mask)

            logits = self.model.forward(input_ids[...,-1:], past_key_values, loras=loras, input_mask=attention_mask).to(input_ids.device)
        else:
            if past_key_values is None:
                past_key_values = ExLlamaV2Cache(self.model, input_ids.shape[0], cache_size)

            logits = self.model.forward(input_ids, past_key_values, loras=loras, input_mask=attention_mask)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, past_key_values if use_cache else None)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values if use_cache else None, loss=loss)

    def load_adapter(self, lora_path: Union[str, os.PathLike], adapter_name: str):

        if adapter_name in self.loras:
            raise ValueError('This adapter is already existed')

        if isinstance(lora_path, str):
            lora_path = Path(lora_path)

        lora_model = ExLlamaV2Lora.from_directory(self.model, lora_path)

        self.loras[adapter_name] = lora_model

    def set_adapter(self, adapter_name: str):

        if adapter_name not in self.loras:
            raise ValueError('The adapter is not existed')

        self._active_adapter = adapter_name

    def enable_adapter_layers(self):

        self._adapter_enabled = True

    def disable_adapter_layers(self):

        self._adapter_enabled = False

    @contextmanager
    def disable_adapter(self):

        try:
            self.disable_adapter_layers()
            yield
        finally:
            self.enable_adapter_layers()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        gpu_split: Optional[str] = None,
        lora_path: Optional[Union[str, os.PathLike]] = None,
        adapter_name: str = 'default',
        trust_remote_code: bool = False,
        use_flash_attention_2: bool = False
    ):
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if isinstance(lora_path, str):
            lora_path = Path(lora_path)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        try:
            generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        except:
            generation_config = GenerationConfig()

        exllama_config = ExLlamaV2Config()
        exllama_config.model_dir = pretrained_model_name_or_path
        exllama_config.no_flash_attn = not use_flash_attention_2
        if getattr(config, 'rope_scaling', None) is not None:
            if config.rope_scaling['type'] == 'linear':
                exllama_config.scale_pos_emb = config.rope_scaling['factor']
            elif config.rope_scaling['type'] == 'dynamic':
                exllama_config.scale_alpha_value = config.rope_scaling['factor']
        exllama_config.prepare()

        model = ExLlamaV2(exllama_config)
        if gpu_split is not None:
            gpu_split = [float(d) for d in gpu_split.split(' ')]
        model.load(gpu_split=gpu_split)

        lora_model = None
        if lora_path is not None:
            lora_model = ExLlamaV2Lora.from_directory(model, lora_path)

        if lora_model is None:
            adapter_name = ''

        return cls(config, generation_config, exllama_config, model, {adapter_name: lora_model}, adapter_name)

    @staticmethod
    def _reorder_cache(past_key_values: ExLlamaV2Cache, beam_idx):

        for i in range(len(past_key_values.key_states)):
            past_key_values.key_states[i] = past_key_values.key_states[i].index_select(0, beam_idx.to(past_key_values.key_states[i].device))
            past_key_values.value_states[i] = past_key_values.value_states[i].index_select(0, beam_idx.to(past_key_values.value_states[i].device))

        return past_key_values
import torch
import time
from transformers import AutoTokenizer

# load the model and tokenizer
# model = ExLlamaV2ForCausalLM.from_pretrained('Llama-2-7b-chat', use_flash_attention_2=True)
model_dir = '/workspace/capybarahermes-2.5-gptq/TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ/'
model = ExLlamaV2ForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# make batch of text with same input
text = ['[INST] What is AI? [/INST]', '[INST] What is AI? [/INST]']
text = text[:1]

inputs = tokenizer(text, return_tensors='pt')

start = time.time()
# make generation deterministic
with torch.inference_mode():
    outputs = model.generate(**inputs, do_sample=False)

end = time.time()
latency = (end - start)

print(f"took {latency} ")

print(tokenizer.batch_decode(outputs))