"""Isolated official Quilt inference for an exact native CapabilityRequest prompt."""
import argparse,json,sys,time,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    request=json.load(sys.stdin)
    assert not a.output.exists()
    sys.path.insert(0,'/home/dbw/merit-feddg-pathology-quilt/scripts')
    from run_quilt_worker import assert_single_gpu,suffix_tokens
    sys.path.insert(0,'/home/dbw/.runtime/quilt-llava-official')
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates,SeparatorStyle
    from llava.mm_utils import process_images,tokenizer_image_token,KeywordsStoppingCriteria
    from PIL import Image
    assert_single_gpu(torch,request['gpu_uuid']);torch.set_num_threads(4);torch.manual_seed(42)
    assert hashlib.sha256(Path(request['image']).read_bytes()).hexdigest()==request['image_sha256']
    started=time.perf_counter()
    tokenizer,model,processor,context_len=load_pretrained_model(request['checkpoint'],None,
        'Quilt-Llava-v1.5-7b',device_map={'':0},device='cuda:0')
    model.eval().requires_grad_(False);loaded=time.perf_counter()
    with Image.open(request['image']) as im:pixels=process_images([im.convert('RGB')],processor,model.config)
    pixels=pixels.to('cuda:0',dtype=torch.float16)
    c=conv_templates['llava_v1'].copy();c.append_message(c.roles[0],'<image>\n'+request['prompt']);c.append_message(c.roles[1],None)
    ids=tokenizer_image_token(c.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to('cuda:0')
    expanded=ids.shape[1]+model.get_vision_tower().num_patches-1
    assert expanded+request['max_new_tokens']<=min(context_len,model.config.max_position_embeddings)
    stop=c.sep if c.sep_style!=SeparatorStyle.TWO else c.sep2
    with torch.inference_mode():
        seq=model.generate(ids,images=pixels,do_sample=False,num_beams=1,use_cache=True,
            max_new_tokens=request['max_new_tokens'],stopping_criteria=[KeywordsStoppingCriteria([stop],tokenizer,ids)])
    tokens=suffix_tokens(seq[0].tolist(),ids[0].tolist());text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
    if stop and text.endswith(stop):text=text[:-len(stop)].strip()
    if not text or len(tokens)>=request['max_new_tokens']:raise RuntimeError('empty/capped native expert output')
    output={'text':text,'input_tokens':int(ids.shape[1]),'output_tokens':len(tokens),'token_ids':tokens}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as h:json.dump({'request':request,'output':output,'load_seconds':loaded-started,
        'inference_seconds':time.perf_counter()-loaded,'expanded_input_tokens':int(expanded)},h)

if __name__=='__main__':main()
