import argparse
import json
import os
from functools import partial
import torch
from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
import tensorrt as trt
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ModelConfig, SamplingConfig, Session,
                                  TensorInfo)

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import requests
import pickle
import datetime
import time

from internvl.train.dataset import build_transform, dynamic_preprocess

def trt_dtype_to_torch(dtype):
    """Convert TensorRT dtype to PyTorch dtype"""
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    elif dtype == trt.DataType.HALF:
        return torch.float16
    elif dtype == trt.DataType.INT32:
        return torch.int32
    elif dtype == trt.DataType.BOOL:
        return torch.bool
    elif dtype == trt.DataType.INT8:
        return torch.int8
    elif dtype == trt.DataType.UINT8:
        return torch.uint8
    elif dtype == trt.DataType.INT64:
        return torch.int64
    elif dtype == trt.DataType.BF16:
        return torch.bfloat16
    else:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")

class TRTCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, input_size=448, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        """
        Dataset class for TensorRT inference
        Args:
            root: Root directory containing images
            annotation: Path to annotation file
            input_size: Input image size
            dynamic_image_size: Whether to use dynamic image size
            use_thumbnail: Whether to use thumbnail
            max_num: Maximum number of images
        """
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
        prompt = "<image> \n" + "Please respond in English. \n" + data_item['conversations'][0]["value"]
        image_id = data_item['id']
        image_path = os.path.join(self.root, data_item['image'])
        
        # Load and preprocess image
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
            'gt': data_item['conversations'][1]["value"],
            'image_path': image_path
        }

class InternInfer(object):

    def __init__(
        self,
        tokenizer_dir,
        intern_engine_dir,
        log_level,
        num_beams,
        vit_engine_path,
    ):
        self.tokenizer_dir = tokenizer_dir
        self.intern_engine_dir = intern_engine_dir
        self.log_level = log_level
        self.global_max_input_len = 2048
        self.decoder = None
        self.tokenizer = None
        self.config = None
        self.sampling_config = None
        self.num_beams = num_beams
        self.model_config = None
        self.vit_engine_path = vit_engine_path
        self.vit_session = None

    def debug_model_config(self):
        """Debug function to print model configuration"""
        print("\nModel Configuration:")
        print(f"Max batch size: {self.model_config.max_batch_size}")
        print(f"Hidden size: {self.model_config.hidden_size}")
        print(f"Vocab size: {self.model_config.vocab_size}")
        print(f"Num layers: {self.model_config.num_layers}")
        print(f"Num heads: {self.model_config.num_heads}")
        print(f"Num kv heads: {self.model_config.num_kv_heads}")
        print(f"Dtype: {self.model_config.dtype}")
        print(f"Remove input padding: {self.model_config.remove_input_padding}")
        print(f"Use gpt attention plugin: {self.model_config.gpt_attention_plugin}")
        print(f"Quant mode: {self.model_config.quant_mode}")
        print(f"Tokens per block: {self.model_config.tokens_per_block}")
        print(f"Max prompt embedding table size: {self.model_config.max_prompt_embedding_table_size}")
        print(f"Max beam width: {self.model_config.max_beam_width}")

    def validate_inputs(self, input_ids, input_lengths, max_new_tokens):
        """Validate inputs before generation"""
        # print("\nValidating inputs:")
        
        # Check input dimensions
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        # Validate against model config
        if batch_size > self.model_config.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.model_config.max_batch_size}")
        
        # Check max length
        total_length = seq_length + max_new_tokens
        if total_length > self.global_max_input_len:
            raise ValueError(f"Total sequence length {total_length} exceeds maximum {self.global_max_input_len}")
        
        # Validate input lengths
        max_length = input_lengths.max().item()
        if max_length != seq_length:
            raise ValueError(f"Max input length {max_length} doesn't match sequence dimension {seq_length}")

    def get_model(self):
        # --load the tokenizer and engine #
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            legacy=False,
            trust_remote_code=True,
        )
        config_path = os.path.join(self.intern_engine_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        gen_config_path = os.path.join(self.tokenizer_dir,
                                       "generation_config.json")
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)
        top_k = 50
        top_p = 1.0
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

        use_gpt_attention_plugin = config["build_config"]["plugin_config"][
            "gpt_attention_plugin"]
        remove_input_padding = config["build_config"]["plugin_config"][
            "remove_input_padding"]
        dtype = config["pretrained_config"]["dtype"]
        tp_size = config["pretrained_config"]["mapping"]["tp_size"]
        pp_size = config["pretrained_config"]["mapping"]["pp_size"]
        world_size = tp_size * pp_size
        assert (
            world_size == tensorrt_llm.mpi_world_size()
        ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
        num_heads = config["pretrained_config"][
            "num_attention_heads"] // world_size
        max_batch_size = config["build_config"]["max_batch_size"]
        hidden_size = config["pretrained_config"]["hidden_size"] // world_size
        vocab_size = config["pretrained_config"]["vocab_size"]
        num_layers = config["pretrained_config"]["num_hidden_layers"]
        num_kv_heads = config["pretrained_config"].get("num_key_value_heads",
                                                       num_heads)
        paged_kv_cache = config["build_config"]["plugin_config"].get("paged_kv_cache", False)

        tokens_per_block = config["build_config"]["plugin_config"][
            "tokens_per_block"]
        max_prompt_embedding_table_size = config["build_config"].get(
            "max_prompt_embedding_table_size", 0)
        quant_mode = QuantMode.from_quant_algo(
            config["pretrained_config"]["quantization"]["quant_algo"],
            config["pretrained_config"]["quantization"]["kv_cache_quant_algo"],
        )
        if config["pretrained_config"].get("multi_query_mode", False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=runtime_rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        model_config = ModelConfig(
            max_batch_size=max_batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            # kv_cache_type=kv_cache_type,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            remove_input_padding=remove_input_padding,
            dtype=dtype,
            quant_mode=quant_mode,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            max_beam_width=self.num_beams,
        )

        sampling_config = SamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=self.num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=0.3,
        )

        engine_name = f"rank{runtime_rank}.engine"
        serialize_path = os.path.join(self.intern_engine_dir, engine_name)
        print(f"Loading engine from {serialize_path}")
        return (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            eos_token_id,
            pad_token_id,
        )

    def intern_model_init(self):
        (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            eos_token_id,
            pad_token_id,
        ) = self.get_model()
        
        # 初始化decoder
        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()
        self.decoder = tensorrt_llm.runtime.GenerationSession(
            model_config,
            engine_buffer,
            runtime_mapping,
        )
        
        # 初始化其他属性
        self.tokenizer = tokenizer
        self.sampling_config = sampling_config
        self.model_config = model_config
        self.config, _ = AutoConfig.from_pretrained(
            self.tokenizer_dir,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )

        # 初始化ViT session
        logger.info(f"Loading ViT engine from {self.vit_engine_path}")
        with open(self.vit_engine_path, "rb") as f:
            vit_engine_buffer = f.read()
        logger.info(f"Creating ViT session")
        self.vit_session = Session.from_serialized_engine(vit_engine_buffer)

    def ptuning_setup(self, prompt_table, dtype, hidden_size, tasks, input_ids):
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                           dtype=torch.int32,
                                           device="cuda")
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))
            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(",")],
                                 dtype=torch.int32,
                                 device="cuda")
            assert (tasks.shape[0] == input_ids.shape[0]
                    ), "Number of supplied tasks must match input batch size"
        else:
            tasks = torch.zeros([input_ids.size(0)], dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def make_context(
            self,
            query: str,
            # system: str = "You are a helpful assistant.",
            system: str = "You are InternVL, a multimodal large language model developed jointly by Shanghai AI Laboratory and SenseTime. You are a helpful and harmless AI assistant.",
            max_window_size: int = 6144,
        ):
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [151644]
            im_end_tokens = [151645]
            nl_tokens = self.tokenizer.encode("\n")

            def _tokenize_str(role, content):
                if "<image>" in content:
                    content = content.replace("<image>", "<img>"+"<IMG_CONTEXT>"*256+"</img>")
                return f"{role}\n{content}", self.tokenizer.encode(role) + \
                    nl_tokens + self.tokenizer.encode(content, add_special_tokens=True)

            system_text, system_tokens_part = _tokenize_str("system", system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            raw_text = ""
            context_tokens = []

            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (nl_tokens + im_start_tokens +
                            _tokenize_str("user", query)[1] + im_end_tokens +
                            nl_tokens + im_start_tokens +
                            self.tokenizer.encode("assistant") + nl_tokens)
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

            return raw_text, context_tokens

    def generate_for_internvl(
        self,
        input_tokens,
        max_new_tokens: int,
        prompt_table=None,
        tasks=None,
        task_vocab_size=None,
        num_beams=1,
    ):
        input_ids = torch.as_tensor(input_tokens,
                                    device="cuda",
                                    dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(max_new_tokens,
                             self.global_max_input_len - max_input_length)

        profiler.start("Intern")
        run_time = 1
        for _ in range(run_time):
            self.validate_inputs(input_ids, input_lengths, max_new_tokens)
            self.decoder.setup(
                batch_size=input_lengths.size(0),
                max_context_length=max_input_length,
                max_new_tokens=max_new_tokens,
                beam_width=num_beams,
            )
            output_ids = self.decoder.decode(
                input_ids,
                input_lengths,
                self.sampling_config,
                prompt_table,
                tasks,
                task_vocab_size,
            )
            torch.cuda.synchronize()
        profiler.stop("Intern")
        Intern_time = profiler.elapsed_time_in_sec("Intern") / run_time

        return output_ids, Intern_time

    def intern_infer(
        self,
        image_embeds,
        image_paths,
        input_text,
        max_new_tokens,
        num_beams=1,
    ):
        if image_paths is None:
            content_list = []
        else:
            content_list = image_paths
        content_list.append({"text": input_text})
        query = input_text
        raw_text, context_tokens = self.make_context(query)
        input_ids = torch.tensor([context_tokens]).to("cuda")

        bos_pos = torch.where(input_ids == 151646)
        eos_pos = torch.where(input_ids == 151647)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        vocab_size = self.config.llm_config.vocab_size

        fake_prompt_id = torch.arange(
            vocab_size,
            vocab_size + image_embeds.shape[0] * image_embeds.shape[1],
            device="cuda",
        )
        fake_prompt_id = fake_prompt_id.reshape(image_embeds.shape[0],
                                                image_embeds.shape[1])
        for idx, (i, a, b) in enumerate(img_pos):
            input_ids[i][a + 1:b] = fake_prompt_id[idx]
        input_ids = input_ids.contiguous().to(torch.int32).cuda()
        input_lengths = torch.tensor(input_ids.size(1),
                                     dtype=torch.int32).cuda()

        dtype = self.model_config.dtype
        prompt_table, tasks, task_vocab_size = self.ptuning_setup(
            image_embeds, dtype, self.model_config.hidden_size, None, input_ids)

        output_ids, Intern_time = self.generate_for_internvl(
            input_ids,
            max_new_tokens,
            prompt_table, 
            tasks,
            task_vocab_size,
            num_beams
        )

        runtime_rank = tensorrt_llm.mpi_rank()
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
        effective_output_token = 0
        if runtime_rank == 0:
            for b in range(input_lengths.size(0)):
                inputs = input_ids[b]
                if content_list is not None:
                    print(f'Input: "{content_list}"')
                    print("\n")
                if self.num_beams <= 1:
                    outputs = output_ids[b][0, len(inputs):].tolist()
                    # outputs = output_ids[b][0, :].tolist()
                    try:
                        effective_output_token = (effective_output_token +
                                                    outputs.index(151643))
                    except:
                        effective_output_token = 1
                    output_text = self.tokenizer.decode(
                        outputs, skip_special_tokens=True)
                    print(f'Output: "{output_text}"')
                    print("\n")
                else:
                    for beam in range(self.num_beams):
                        outputs = output_ids[b][beam, len(inputs):].tolist()
                        output_text = self.tokenizer.decode(
                            outputs, skip_special_tokens=True)
                        print(f'Output(beam: {beam}): "{output_text}"')
        logger.info(f"Input length={input_lengths[b]}")
        logger.info(f"Output length={output_ids.shape}")
        logger.info(f"TensorRT-LLM Intern time: {Intern_time:3f} sec ")
        return output_text

    def process_image(self, pixel_values, stream):
        """
        使用ViT处理图像
        Args:
            pixel_values: 预处理后的图像张量 [B, C, H, W]
            stream: CUDA流
        Returns:
            image_embeds: ViT输出的图像嵌入
        """
        visual_inputs = {"input": pixel_values.bfloat16()}
        visual_output_info = self.vit_session.infer_shapes(
            [TensorInfo("input", trt.DataType.BF16, pixel_values.shape)])
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                              dtype=trt_dtype_to_torch(t.dtype),
                              device="cuda")
            for t in visual_output_info
        }
        
        profiler.start("ViT")
        ok = self.vit_session.run(visual_inputs, visual_outputs, stream)
        profiler.stop("ViT")
        vit_time = profiler.elapsed_time_in_sec("ViT")
        logger.info(f"TensorRT-LLM ViT latency: {vit_time:3f} sec")

        assert ok, "Runtime execution failed for vit session"
        return visual_outputs["output"]

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
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

def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')
    gts = [_['gt'] for _ in inputs]
    image_paths = [_['image_path'] for _ in inputs]

    return pixel_values, image_ids, input_tokens.input_ids, input_tokens.attention_mask, gts, image_paths


def main():
    args = parse_arguments()
    
    # Set max_new_tokens based on mode
    max_new_tokens = 300 if args.mode == "slow" else 10
    
    # Create log directory if it doesn't exist
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = log_path / f'predictions_{timestamp}.json'
    
    # Initialize empty results file if it doesn't exist
    if not output_file.exists():
        with open(output_file, 'w') as f:
            json.dump([], f)

    qinfer = InternInfer(
        args.tokenizer_dir,
        args.intern_engine_dir,
        args.log_level,
        args.num_beams,
        args.vit_engine_path,
    )
    qinfer.intern_model_init()
    qinfer.debug_model_config()

    dataset = TRTCaptionDataset(
        root=args.root,
        annotation=args.data_file,
        input_size=args.input_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=False,  # 可以根据需要设置为参数
        max_num=args.max_num
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=qinfer.tokenizer),
    )

    for batch_idx, (pixel_values, ids, input_ids, attention_mask, gts, image_paths) in tqdm(enumerate(dataloader)):
        print("\n" + "="*50)
        print(f"Processing batch {batch_idx}, image paths: {image_paths}")
        
        stream = torch.cuda.current_stream().cuda_stream
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        image_embeds = qinfer.process_image(pixel_values, stream)
        output_text = qinfer.intern_infer(
            image_embeds,
            image_paths,
            dataset[batch_idx]['input_text'],
            max_new_tokens,
            args.num_beams,
        )
        end_time2 = time.time()
        # Read existing results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Append new results
        for img_path, gt, pred in zip(image_paths, gts, [output_text]):
            results.append({
                'image_path': img_path,
                'ground_truth': gt,
                'prediction': pred
            })

        # Write updated results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Ground Truth: {gts} \n")
        print(f"Prediction: {output_text} \n")
        print("="*50 + "\n")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="slow", choices=["slow", "fast"],
                       help="Inference mode: 'slow' for max_new_tokens=300, 'fast' for max_new_tokens=10")
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--vit_engine_path", type=str, required=True)
    parser.add_argument("--intern_engine_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument('--dynamic', action='store_true', help='Whether to use dynamic image size')
    parser.add_argument('--max-num', type=int, default=6, help='Maximum number of images')
    parser.add_argument('--input-size', type=int, default=448, help='Input image size')
    
    return parser.parse_args()

if __name__ == "__main__":
    main()