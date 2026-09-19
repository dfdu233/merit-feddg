"""Frozen official LLaVA-Critic inference; no training or shared dependency changes.

Source: LLaVA-VL/LLaVA-NeXT and lmms-lab/llava-critic-7b model-card example.
The finite-choice interface is our explicit adaptation, not a paper reproduction.
"""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path('/home/dbw/merit-feddg/artifacts/upstream/LLaVA-NeXT-critic-compat')
CHECKPOINT = Path('/home/dbw/models/llava-critic-7b-498f2d7')
REVISION = '498f2d719b83e50e48787c6958afe7100503c23f'
WEIGHTS = {
    'model-00001-of-00004.safetensors': 'a3c2dac0e595e5b40072a604ff401df2e781d702a42d3175a7e6737aaa44d437',
    'model-00002-of-00004.safetensors': '0b30c8b75a6c81a454f0e3eae6c4106d7a6cd79b17b55f24c57c440cd6ab4703',
    'model-00003-of-00004.safetensors': 'fd3b0171bd7dca40dc4cff06524dc5b9979ef0f4905a1084be0768ae1c8cd316',
    'model-00004-of-00004.safetensors': '6611121dc8a845f7b85d003b76bc2cf59ad4f2938099e23ce82072d69bacb5a6',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def identity():
    index = json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())
    weights = sorted(set(index['weight_map'].values()))
    hashes = {}
    if set(weights) != set(WEIGHTS):
        raise ValueError('Unexpected checkpoint shard index')
    for name in weights:
        digest = sha(CHECKPOINT/name)
        if digest != WEIGHTS[name]:
            raise ValueError('Pinned HF LFS SHA256 mismatch: '+name)
        hashes[str(CHECKPOINT/name)] = digest
    source = {str(p): sha(p) for p in sorted((UPSTREAM/'llava').rglob('*.py'))}
    source[str(Path(__file__).resolve())] = sha(__file__)
    source['constrained'] = sha(Path(__file__).resolve().parents[1]/'merit_feddg/constrained.py')
    return {'name': 'lmms-lab/llava-critic-7b', 'revision': REVISION,
        'vision_source': 'all 421 visual parameters embedded in critic checkpoint', 'weights_sha256': hashes,
        'upstream_commit': subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'], text=True).strip(),
        'source_sha256': source, 'checkpoint_config_sha256': sha(CHECKPOINT/'config.json'),
        'dtype': 'bfloat16', 'attention': 'eager', 'template': 'qwen_1_5',
        'training': False, 'decision_channel': 'finite_choice',
        'note': 'Closed-label adaptation; not official free-form critic reproduction'}


class LlavaCritic:
    def __init__(self):
        sys.path.insert(0, str(UPSTREAM))
        import torch
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
        from transformers import AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), local_files_only=True)
        config = LlavaQwenConfig.from_pretrained(str(CHECKPOINT), local_files_only=True)
        # Legacy export contains a metadata-only Llama stub, not a decoder.
        # Native LLaVA-Qwen uses top-level Qwen2 parameters; modern Transformers
        # otherwise mistakes this dict for a nested PretrainedConfig.
        if getattr(config, 'text_config', None) != {'model_type': 'llama'}:
            raise ValueError('Unexpected legacy text_config; inspect before adapting')
        del config.text_config
        config.vision_weights_in_main_checkpoint = True
        self.model, info = LlavaQwenForCausalLM.from_pretrained(
            str(CHECKPOINT), config=config, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Critic weight loading is not exact: '+repr(info))
        original_rows = self.model.get_input_embeddings().weight.shape[0]
        if original_rows < len(self.tokenizer):
            raise RuntimeError('Do not randomly initialize new critic embeddings')
        # Official builder trims Qwen padding rows to the actual tokenizer size.
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval().requires_grad_(False)
        self.model.get_vision_tower().to(device='cuda:0', dtype=torch.bfloat16)
        self.image_processor = self.model.get_vision_tower().image_processor
        self.context_limit = min(int(config.max_position_embeddings), int(config.tokenizer_model_max_length))
        self.loading_info = dict(info, original_vocab_rows=original_rows,
                                 retained_vocab_rows=len(self.tokenizer))

    def _inputs(self, image_path, prompt):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from PIL import Image
        with Image.open(image_path) as f:
            image = f.convert('RGB')
        conversation = copy.deepcopy(conv_templates['qwen_1_5'])
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN+'\n'+prompt)
        conversation.append_message(conversation.roles[1], None)
        ids = tokenizer_image_token(conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt').unsqueeze(0).to('cuda:0')
        pixels = [x.to(device='cuda:0', dtype=self.torch.bfloat16)
                  for x in process_images([image], self.image_processor, self.model.config)]
        if int((ids == IMAGE_TOKEN_INDEX).sum()) != 1 or len(pixels) != 1:
            raise ValueError('Expected one original image, with native anyres crops')
        tiles = sum(x.shape[0] if x.ndim == 4 else 1 for x in pixels)
        # Upper bound includes patch and row-newline tokens; no silent truncation.
        patches = int(self.model.get_vision_tower().num_patches)
        upper = int(ids.shape[1])-1 + tiles*(patches+int(patches**0.5)+1)
        return ids, pixels, image.size, upper

    def context_token_budget(self, image, prompt, reserve_tokens):
        _, _, _, upper = self._inputs(image, prompt)
        return {'fits': upper+reserve_tokens <= self.context_limit,
                'input_tokens_upper_bound': upper, 'context_limit': self.context_limit}

    def generate_with_usage(self, image, prompt, max_new_tokens=8, *, allowed_texts=None):
        import merit_feddg.constrained as constraint_module
        from merit_feddg.constrained import finite_choice_constraint
        expected = Path(__file__).resolve().parents[1]/'merit_feddg/constrained.py'
        if sha(constraint_module.__file__) != sha(expected):
            raise ValueError('Runtime finite-choice helper differs from recorded source')
        ids, pixels, size, upper = self._inputs(image, prompt)
        if upper+max_new_tokens > self.context_limit:
            raise ValueError('Critic context unavailable; do not truncate answers or image')
        options = {}
        if allowed_texts:
            options['prefix_allowed_tokens_fn'] = finite_choice_constraint(
                self.tokenizer, allowed_texts, eos_token_id=self.tokenizer.eos_token_id)
        with self.torch.inference_mode():
            result = self.model.generate(ids, images=pixels, image_sizes=[size],
                attention_mask=self.torch.ones_like(ids),
                do_sample=False, num_beams=1, use_cache=True, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id, **options)
        if any(self.torch.isnan(s).any() or self.torch.isposinf(s).any() for s in result.scores):
            raise ValueError('Nonfinite critic logits; do not treat as a medical verdict')
        token_ids = result.sequences[0].tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if allowed_texts and text not in allowed_texts:
            raise ValueError('Critic failed finite grammar, no implicit KEEP')
        return {'text': text, 'token_ids': token_ids, 'output_tokens': len(token_ids),
                'input_tokens_upper_bound': upper, 'image_tensor_shapes': [list(x.shape) for x in pixels]}
