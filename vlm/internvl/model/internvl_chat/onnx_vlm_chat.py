import onnx
import onnxruntime
import torch
import numpy as np
import warnings
class OnnxVlmChatModel():
    def __init__(self, config, language_model_path, vision_model_path, mlp1_path, embeds_weight_path):
        self.language_model_path = language_model_path
        self.vision_model_path = vision_model_path
        self.mlp1_path = mlp1_path
        self.language_session = onnxruntime.InferenceSession(
            language_model_path, 
            providers=['CUDAExecutionProvider']
        )
        self.vision_session = onnxruntime.InferenceSession(
            vision_model_path, 
            providers=['CUDAExecutionProvider']
        )
        self.mlp1_session = onnxruntime.InferenceSession(
            mlp1_path, 
            providers=['CUDAExecutionProvider']
        )
        self.embeds_weight = np.load(embeds_weight_path)
        self.device = getattr(config, 'device', 'cpu')
        self.ps_version = config.ps_version

        self.hidden_size = config.llm_config.hidden_size
        self.vocab_size = config.llm_config.vocab_size

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def get_input_embeddings(self, input_ids):
        return self.embeds_weight[input_ids]

    def forward_language(self, input_ids, attention_mask, inputs_embeds):
        return self.language_session.run(
            None, 
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'inputs_embeds': inputs_embeds
            }
        )
    
    def forward_vision(self, pixel_values):
        return self.vision_session.run(
            None, 
            {
                'pixel_values': pixel_values.to(torch.float16).to(self.device).numpy()
            }
        )
    
    def forward_mlp1(self, vit_embeds):
        return self.mlp1_session.run(
            None, 
            {
                'input': vit_embeds.to(torch.float16).to(self.device).numpy()
            }
        )

    def extract_feature(self, pixel_values):
        vit_embeds = self.forward_vision(pixel_values)[0]
        vit_embeds = torch.from_numpy(vit_embeds) # ndarray -> tensor
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.forward_mlp1(vit_embeds)[0]
        return vit_embeds
    
    def generate_language(self, input_ids, attention_mask):
        """
        generate the text based on the language model
        """
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        for _ in range(self.generation_config.max_new_tokens):
            language_outputs = self.forward_language(current_input_ids, current_attention_mask)
            next_token_logits = language_outputs[0][:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)

            if next_token == self.generation_config.eos_token_id:
                break

            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1))], dim=-1)

        return current_input_ids

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        