"""Bounded real-output batching comparison; never modifies production caches."""
import argparse,json,sys,time
from pathlib import Path
import native_quilt_infer as native

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    jobs=json.loads(a.jobs.read_text())[:4]
    assert len(jobs)==4
    # Load/warm with the same official backend, then measure four serial requests.
    native.run(jobs[0]['request'],a.output/'warm.json')
    import torch
    from PIL import Image
    from transformers import LogitsProcessor,LogitsProcessorList,StoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates,SeparatorStyle
    from llava.mm_utils import process_images,tokenizer_image_token,KeywordsStoppingCriteria
    tok,model,processor,context=native._loaded[1]
    torch.cuda.synchronize();started=time.perf_counter()
    for i,j in enumerate(jobs):native.run(j['request'],a.output/f'serial-{i}.json')
    torch.cuda.synchronize();serial_seconds=time.perf_counter()-started
    actual=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    for offset in (0,2):
        pair=jobs[offset:offset+2];inputs=[];pixels=[];stops=[]
        for j in pair:
            r=j['request'];conv=conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0],'<image>\n'+r['prompt']);conv.append_message(conv.roles[1],None)
            inputs.append(tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors='pt'))
            with Image.open(r['image']) as im:pixels.append(im.convert('RGB'))
            stops.append(conv.sep if conv.sep_style!=SeparatorStyle.TWO else conv.sep2)
        assert len({j['request']['max_new_tokens'] for j in pair})==1
        cap=pair[0]['request']['max_new_tokens'];width=max(len(x) for x in inputs)
        assert width+model.get_vision_tower().num_patches-1+cap<=min(context,model.config.max_position_embeddings)
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids=torch.full((2,width),pad,dtype=torch.long,device='cuda:0');mask=torch.zeros_like(ids)
        for i,x in enumerate(inputs):ids[i,-len(x):]=x.to('cuda:0');mask[i,-len(x):]=1
        criteria=[KeywordsStoppingCriteria([stops[i]],tok,ids[i:i+1]) for i in range(2)]
        lengths=[None,None]
        class Finish(StoppingCriteria):
            def __call__(self,out,scores,**kwargs):
                for i in range(2):
                    if lengths[i] is None and criteria[i](out[i:i+1],scores):lengths[i]=out.shape[1]-width
                return all(v is not None for v in lengths)
        class KeepFinished(LogitsProcessor):
            def __call__(self,out,scores):
                for i,n in enumerate(lengths):
                    if n is not None:scores[i,:]=-float('inf');scores[i,tok.eos_token_id]=0
                return scores
        torch.manual_seed(42)
        with torch.inference_mode():
            seq=model.generate(ids,attention_mask=mask,images=process_images(pixels,processor,model.config).to('cuda:0',dtype=torch.float16),
                do_sample=False,num_beams=1,use_cache=True,max_new_tokens=cap,
                stopping_criteria=[Finish()],logits_processor=LogitsProcessorList([KeepFinished()]))
        assert torch.equal(seq[:,:width],ids)
        for i in range(2):
            tokens=seq[i,width:width+(lengths[i] if lengths[i] is not None else cap)].tolist()
            text=tok.decode(tokens,skip_special_tokens=True).strip()
            if stops[i] and text.endswith(stops[i]):text=text[:-len(stops[i])].strip()
            ref=json.loads((a.output/f'serial-{offset+i}.json').read_text())['output']
            actual.append({'index':offset+i,'token_ids':tokens,'text':text,'token_exact':tokens==ref['token_ids'],'text_exact':text==ref['text']})
    torch.cuda.synchronize();batch_seconds=time.perf_counter()-started
    result={'serial_seconds':serial_seconds,'batch2_seconds':batch_seconds,'speedup':serial_seconds/batch_seconds,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cases':actual,
            'note':'Bounded engineering comparison under concurrent production workload; not a quality score or universal equivalence proof.'}
    with (a.output/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    print('TOKEN_EXACT',sum(x['token_exact'] for x in actual),'TEXT_EXACT',sum(x['text_exact'] for x in actual),flush=True)

if __name__=='__main__':main()
