import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import numpy as np
import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from statistics import mean

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        with open(annotation, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        prompt = "<image> \n" + data_item['conversations'][0]["value"]
        image_id = data_item['id']
        image_path = os.path.join(self.root, data_item['image'])
        print()
        print("------------------------------", image_path, "------------------------------")
        print(data_item['conversations'][1]["value"])
        print()
        image = Image.open(image_path)
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'image_id': image_id,
            'input_text': prompt,
            'pixel_values': pixel_values,
            'gt': data_item['conversations'][1]["value"]
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')
    gts = [_['gt'] for _ in inputs]

    return pixel_values, image_ids, input_texts[0], input_tokens.attention_mask, gts


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model(args):
    random.seed(args.seed)

    dataset = CaptionDataset(
        root=args.root,
        annotation=args.data_file,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'eval_{time_prefix}.json'
    results_file = os.path.join(args.out_dir, results_file)
    
    if torch.distributed.get_rank() == 0:
        # 创建文件并写入开始的方括号
        with open(results_file, 'w') as f:
            f.write('[\n')
    
    image_ids, captions, gt_actions, times_used = [], [], [], []
    total_frames = 0
    
    is_first_result = True
    current_idx = 0
    
    for _, (pixel_values, ids, input_text, _, gts) in tqdm(enumerate(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        print('input_text:', input_text)
        generation_config = dict(
            num_beams=args.num_beams,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=10 if args.mode == "fast" else 100,
        )
        current_time = time.time()
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=input_text,
            generation_config=generation_config,
            verbose=True
        )
        inference_time = time.time() - current_time
        times_used.append(inference_time)
        total_frames += pixel_values.size(0)

        if torch.distributed.get_rank() == 0:
            result = {
                'image_id': int(ids[0]),
                'image_path': dataset.data[current_idx]['image'],
                'ground_truth': gts[0],
                'prediction': pred
            }
            
            with open(results_file, 'a') as f:
                if not is_first_result:
                    f.write(',\n')
                json.dump(result, f, indent=2)
                is_first_result = False
            
            current_idx += 1
        
        image_ids.extend(ids)
        captions.extend([pred])
        gt_actions.extend(gts)
        times_used.append(inference_time)
        total_frames += pixel_values.size(0)

    torch.distributed.barrier()
    
    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_captions = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_captions, captions)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
    average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
    print(f'Average caption length: {average_length}')

    if torch.distributed.get_rank() == 0:
        results = []
        
        for i, (image_id, caption, gt) in enumerate(zip(merged_ids, merged_captions, gt_actions)):
            results.append({
                'image_id': int(image_id),
                'image_path': dataset.data[i]['image'],
                'ground_truth': gt,
                'prediction': caption
            })

        avg_time = mean(times_used)
        fps = 1.0 / avg_time
        
        metrics = {
            'metrics': {
                'average_inference_time': avg_time,
                'fps': fps,
                'total_frames': total_frames
            }
        }
        
        with open(results_file, 'a') as f:
            f.write(',\n')
            json.dump(metrics, f, indent=2)
            f.write('\n]')
        
        print(f"Results saved to {results_file}")
        print(f"Average inference time per frame: {avg_time:.4f} seconds")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames processed: {total_frames}")

    torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='./vlm_eval')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--mode', type=str, default="fast")
    args = parser.parse_args()

    assert args.checkpoint is not None and args.data_file is not None
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    tokenizer.add_eos_token = False
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model(args)