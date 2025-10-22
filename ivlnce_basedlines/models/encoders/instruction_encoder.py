import gzip
import json
import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor
import torch.nn.functional as F
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np

def pad_tokens(token_tensor, target_length=200):
    batch_size, seq_len = token_tensor.shape
    if seq_len > target_length:
        padded_tensor = token_tensor[:, :target_length]
    else:
        padded_tensor = F.pad(token_tensor, (0, target_length - seq_len),
                              'constant', 0)
    return padded_tensor

class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
        self.llava_model = None
        self.fc1 = nn.Linear(3072, 3072)
        self.relu = nn.GELU(approximate='none')
        self.fc2 = nn.Linear(3072, 256 * 12)

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))
    def one_hot_to_rgb(self, one_hot_tensor):
        batch_size, num_classes, height, width = one_hot_tensor.shape
        rgb_tensor = torch.zeros(batch_size, 3, height, width,
                                 dtype=torch.uint8)
        colors = np.array([
            [0, 0, 0],
            [106, 137, 204],
            [230, 126, 34],
            [7, 153, 146],
            [248, 194, 145],
            [76, 209, 55],
            [255, 168, 1],
            [184, 233, 148],
            [39, 174, 96],
            [229, 80, 57],
            [30, 55, 153],
            [24, 220, 255],
            [234, 32, 39]
        ])

        for i in range(batch_size):
            for c in range(num_classes):
                mask = one_hot_tensor[i, c] > 0.5
                rgb_tensor[i, 0][mask] = colors[c, 0]
                rgb_tensor[i, 1][mask] = colors[c, 1]
                rgb_tensor[i, 2][mask] = colors[c, 2]

        return rgb_tensor

    def forward(self, observations: Observations,) -> Tensor:

        instruction = observations["instruction"].long()
        batch_size, seq_len = instruction.size()
        RGB = observations['rgb']
        RGB = RGB.permute(0, 3, 1, 2)
        semantic = observations["semantic_map"].long()
        semantic = F.one_hot(semantic, 13)
        semantic = semantic.permute(0, 3, 1, 2).to(dtype=torch.float)
        semantic_RGB = self.one_hot_to_rgb(semantic).cuda()

        occupancy_map = observations["occupancy_map"]
        occupancy_map = occupancy_map * 255
        occupancy_map = 255 - occupancy_map
        occupancy_map = torch.stack([occupancy_map] * 3, 2)
        occupancy_map = occupancy_map.permute(0, 2, 1, 3).to(dtype=torch.float)
        MAP = 0.5 * occupancy_map + 0.5 * semantic_RGB
        MAP = F.interpolate(MAP, size=(224, 224), mode='bilinear', align_corners=False)

        MAPandRGB = torch.cat((RGB, MAP), dim=2)
        MAPandRGB = F.interpolate(MAPandRGB, size=(384, 384), mode='bilinear', align_corners=False)
        MAPandRGB = torch.round(MAPandRGB).to(torch.int)

        qs = self.llava_processor.batch_decode(instruction,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        qs = [s.replace("!", "") for s in qs]

        My_conversation = []
        for text in qs:
            words = text.split()
            if len(words) > 100:
                words = words[:100]
                text = ' '.join(words)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"Use the current scene and map for navigation. Following the instruction: {text}"},
                        {"type": "image"},
                    ],
                },
            ]
            My_conversation.append(conversation)

        prompt = self.llava_processor.apply_chat_template(My_conversation)

        inputs = self.llava_processor(text=prompt, images=MAPandRGB,
                                      return_tensors="pt",
                                      padding=True).to(0, torch.float16)

        if 'depth' in observations:
            S_batch = 30
            num_splits = (batch_size + S_batch - 1) // S_batch
            all_outputs1 = []
            for i in range(num_splits):
                start_idx = i * S_batch
                end_idx = min((i + 1) * S_batch, batch_size)

                if start_idx == end_idx:
                    continue

                batch_inputs = {k: v[start_idx:end_idx] for k, v in
                                inputs.items()}
                with torch.inference_mode():
                    outputs_all = self.llava_model(**batch_inputs,
                                                output_hidden_states=True).hidden_states
                    middle_layer = outputs_all[-11]
                    last_layer = outputs_all[-1]
                    head_layer = outputs_all[-16]
                    combined_layer = torch.cat(
                        [middle_layer, last_layer, head_layer],
                        dim=-1)
                    combined_layer = combined_layer[:, 747:, :]

                all_outputs1.append(combined_layer)

            outputs1 = torch.cat(all_outputs1, dim=0)

        else:
            S_batch = 6
            num_splits = (batch_size + S_batch-1) // S_batch  # 向上取整
            all_outputs1 = []

            for i in range(num_splits):
                start_idx = i * S_batch
                end_idx = min((i + 1) * S_batch, batch_size)
                if start_idx == end_idx:
                    continue

                batch_inputs = {k: v[start_idx:end_idx] for k, v in
                                inputs.items()}

                with torch.inference_mode():
                    outputs_all = self.llava_model(**batch_inputs,
                                                   output_hidden_states=True).hidden_states
                    middle_layer = outputs_all[-11]
                    last_layer = outputs_all[-1]
                    head_layer = outputs_all[-16]
                    combined_layer = torch.cat([middle_layer, last_layer, head_layer],
                                               dim=-1)

                    combined_layer = combined_layer[:, 747:, :]

                all_outputs1.append(combined_layer)

            outputs1 = torch.cat(all_outputs1, dim=0)

        outputs = outputs1.float()
        _, z, _ = outputs.shape

        if batch_size > 2000:
            S_batch = 2000
            all_outputs = []

            for i in range(0, batch_size, S_batch):
                x_batch = outputs[i:i+S_batch]

                print(x_batch.shape)

                x_batch = self.fc1(x_batch)
                x_batch = self.relu(x_batch)
                x_batch = self.fc2(x_batch)

                all_outputs.append(x_batch)

            outputs = torch.cat(all_outputs, dim=0)
        else:
            outputs = self.fc1(outputs)
            outputs = self.relu(outputs)
            outputs = self.fc2(outputs)

        outputs = outputs.view(batch_size, z, 256, 12)
        outputs_mean = torch.mean(outputs, dim=1)
        outputs_max, _ = torch.max(outputs, dim=1)
        outputs = torch.cat((outputs_mean, outputs_max), dim=2)

        return outputs
