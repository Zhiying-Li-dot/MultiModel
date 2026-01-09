import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb


# https://github.com/TencentARC/MasaCtrl/blob/main/run_synthesis_sdxl_processor.py
def register_attention_processor(
    model: Optional[nn.Module] = None,
    processor_type: str = "MasaCtrlProcessor",
    **attn_args,
):
    """
    Register attention processor to model.

    Args:
        model: a unet model or a list of unet models
        processor_type: the type of the processor
            - "MasaCtrlProcessor": For MasaCtrl cross-attention masking
            - "CleanReferenceAttentionProcessor": For clean product image reference (RefDrop t=0)
            - "NoisyReferenceAttentionProcessor": For noisy reference (RefDrop matching timestep)
            - default: WanAttnProcessor2_0
        **attn_args: Arguments for the processor
    """
    if not isinstance(model, (list, tuple)):
        model = [model]

    if processor_type == "MasaCtrlProcessor":
        # start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_layers=32, total_steps=50, model_type="SD"
        processor = MasaCtrlProcessor(**attn_args)
    elif processor_type == "NoisyReferenceAttentionProcessor":
        # ref_latent, prompt_embeds, transformer, scheduler, c=0.2
        processor = NoisyReferenceAttentionProcessor(**attn_args)
    elif processor_type == "NoisyReferenceAttentionProcessor_restore":
        # Special case: restore existing processor instance
        processor = attn_args.get('processor_instance')
    elif processor_type == "CleanReferenceAttentionProcessor":
        # ref_bank, c=0.2
        # RECOMMENDED: RefDrop with linear interpolation for clean product images
        processor = CleanReferenceAttentionProcessor(**attn_args)
    else:
        processor = WanAttnProcessor2_0()

    for m in model:
        m.set_attn_processor(processor)
        print(f"Model {m.__class__.__name__} is registered attention processor: {processor_type}")


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, None


class MasaCtrlProcessor(nn.Module):
    def __init__(
            self, 
            start_step=4, 
            end_step=40, 
            start_layer=10,
            end_layer=42,
            layer_idx=None, 
            step_idx=None, 
            total_layers=42, 
            total_steps=50, 
            model_type="cogvideo", 
            masking=False, 
            masking_layer = [0],
            frame_num=49,
            source_idx = [0],
            target_idx = [0],
            #source_threshold = 0.55,
            #target_threshold = 0.55
            ):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps, must be same to the denoising steps used in denoising scheduler
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.frame_num = frame_num
        self.total_steps = total_steps
        self.num_attn_layers = total_layers

        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.attn_layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer+1))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, self.end_step+1))

        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at attn layers: ", self.attn_layer_idx)

        self.cur_step = 0
        self.cur_att_layer = 0
        self.masactrl_apply = 0
        self.mask_apply = 0
        self.masking = masking
        self.masking_layer = masking_layer

        self.source_attn_map_list = []
        self.target_attn_map_list = []

        self.source_idx = source_idx
        self.target_idx = target_idx
        self.save_path = './results/mask/'
        os.makedirs(self.save_path, exist_ok=True)

        #self.source_threshold = source_threshold
        #self.target_threshold = target_threshold

        # TODO: check model embedding dimension (6/7)
        self.text_length = 512
        self.video_length = 20280
        self.F_out = frame_num//4 + 1 # flexible
        self.H_out = 30
        self.W_out = 52 
        #self.local_blending_source = None
        #self.local_blending_target = None

    def after_step(self):
        #print("## After step")
        
        ## when use simple masking,
        #self.source_attn_map_list = []
        #self.target_attn_map_list = []

        print("Masking Apply number:", self.mask_apply)
        print("MasaCtrl Apply number:", self.masactrl_apply) # 30
        self.mask_apply = 0
        self.masactrl_apply = 0
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ):

        out = self.attn_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb
        )

        # add multiply 2 -> because self-attn + cross-attn
        #print(self.cur_att_layer, self.num_attn_layers, self.is_cross) # 60, 30
        if self.cur_att_layer == self.num_attn_layers*2:
            self.cur_att_layer = 0
            self.cur_step %= self.total_steps

            # after step
            self.after_step()
            self.cur_step += 1

        return out

    # TODO: visualization attention map
    def attn_map_visualization(
        self,
        attn_map,
        target_flag=False,
        file_save=False,
        image_save=True,
    ):
        
        # pt & gif 저장할 폴더
        gif_name = 'block_'+str((self.cur_att_layer+1)//2)+'.gif'
        file_name = 'block_'+str((self.cur_att_layer+1)//2)+'.pt'
        if target_flag:
            os.makedirs(self.save_path + 'target/step_' + str(self.cur_step+1), exist_ok=True)
            save_name = self.save_path + 'target/step_' + str(self.cur_step+1) + "/" + gif_name
            file_save_name = self.save_path + 'target/step_' + str(self.cur_step+1) + "/" + file_name
        else:
            os.makedirs(self.save_path + 'source/step_' + str(self.cur_step+1), exist_ok=True)
            save_name = self.save_path + 'source/step_' + str(self.cur_step+1) + "/" + gif_name
            file_save_name = self.save_path + 'source/step_' + str(self.cur_step+1) + "/" + file_name
        
        if file_save:
            #print("Attention map file saving...")
            torch.save(attn_map, file_save_name)

        if image_save:
            #print("Attention map image saving...")

            # 임시 이미지 저장 폴더
            temp_dir = "./temp_frames"
            os.makedirs(temp_dir, exist_ok=True)

            # 각 frame 별로 저장
            filenames = []
            for i in range(len(attn_map)):
                fig, ax = plt.subplots()
                ax.imshow(attn_map[i].cpu())
                ax.axis('off')
                filename = os.path.join(temp_dir, f"frame_{i:02d}.png")
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                filenames.append(filename)

            # gif로 저장
            with imageio.get_writer(save_name, mode='I', duration=250) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

            # 임시 파일 삭제
            for filename in filenames:
                os.remove(filename)
            os.rmdir(temp_dir)


    # make attention map
    def make_attn_map(
        self,
        attn_weight
    ) -> torch.Tensor:
        
        '''
        [src_uncond, tar_uncond, src_cond, tar_cond]
        '''
        B, H, L, D = attn_weight.shape

        attn_weight = attn_weight.mean(1) # [B, 20280, 512]

        # TODO: FlowEdit
        #src_attn_weight = attn_weight[2, :, self.source_idx]
        #tar_attn_weight = attn_weight[3, :, self.target_idx]
        
        # TODO: FlowAlign
        src_attn_weight = attn_weight[0, :, self.source_idx]
        tar_attn_weight = attn_weight[2, :, self.target_idx]

        ## 만약 단어가 2개 이상이면
        if len(self.source_idx) >= 2:
            src_attn_weight = src_attn_weight.sum(-1)
            #print(src_attn_weight.shape) # (20280, )
        if len(self.target_idx) >= 2:
            tar_attn_weight = tar_attn_weight.sum(-1)

        ## Reshape
        src_attn_weight = src_attn_weight.reshape(self.F_out, self.H_out, self.W_out)
        tar_attn_weight = tar_attn_weight.reshape(self.F_out, self.H_out, self.W_out)
        #print(tar_attn_weight.shape) # torch.Size([13, 30, 52])

        ## Normalization (each frame)
        #print(src_attn_weight.min(dim=1, keepdim=True)[0].shape) # (13, 1, 52)
        #'''
        src_attn_weight_min = src_attn_weight.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] 
        src_attn_weight_max = src_attn_weight.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        tar_attn_weight_min = tar_attn_weight.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] 
        tar_attn_weight_max = tar_attn_weight.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        
        src_attn_weight = (src_attn_weight - src_attn_weight_min) / (src_attn_weight_max - src_attn_weight_min)
        tar_attn_weight = (tar_attn_weight - tar_attn_weight_min) / (tar_attn_weight_max - tar_attn_weight_min)
        #'''

        ## File save & Visualization
        #self.attn_map_visualization(src_attn_weight, target_flag=False, file_save=True, image_save=False)
        #self.attn_map_visualization(tar_attn_weight, target_flag=True, file_save=True, image_save=False)

        #return (src_attn_weight, tar_attn_weight)

        ## cumulative masking
        self.source_attn_map_list.append(src_attn_weight)
        self.target_attn_map_list.append(tar_attn_weight)

    def attn_forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # TODO: check current layer type
        self.cur_att_layer += 1
        self.is_cross = False if encoder_hidden_states == None else True # check cross attn

        # original attn (below)
        #print(hidden_states.shape) # torch.Size([4, 20280, 1536])
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        #print(query.shape) # torch.Size([4, 20280, 1536])
        #print(key.shape) # torch.Size([4, 20280, 1536]), torch.Size([4, 512, 1536])
        #print(value.shape) # torch.Size([4, 20280, 1536]), torch.Size([4, 512, 1536])

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # TODO: need to add KV-sharing function
        #print("Query shape:", query.shape)
        #print("Key shape:", key.shape)
        #print("Value shape:", value.shape)
        '''
        the batch consists of [uncond_src, uncond_tar, cond_src, cond_tar]

        In self-attn:
        [query, key, value]: torch.Size([4, 12, 20280, 128]) # 12 = attn.heads; 128 = attention dim

        In cross-attn:
        query: torch.Size([4, 12, 20280, 128])
        [key, value]: torch.Size([4, 12, 512, 128]) # text_length = 512
        '''
        # self.attn_layer_idx has range (1,30), but self.cur_att_layer will be 30 upper.
        ## <- 이거 range안에 맞게 들어가도록 if 문을 수정
        ''' # 일단 보류 (6/18)
        if not self.is_cross and ((self.cur_att_layer+1)//2 in self.attn_layer_idx): # mutual self-attn
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            
            self.masactrl_apply += 1
            #print("Not yet perfect. Now just Attention.")

        else: # just attn
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        '''
        hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        # TODO: Cross-Attn 시각화 (6/9)
        if self.is_cross and self.masking and ((self.cur_att_layer+1)//2 in self.masking_layer):
            value_ones = torch.eye(key.shape[2], device=query.device, dtype=query.dtype) # define identity matrix
            attn_weights = F.scaled_dot_product_attention(
                query, key, value_ones, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            #print(attn_weights)
            #print(attn_weights.shape) # [4, 12, 20280, 512]
            '''
            20280 == (13, 30, 52) # (F, H, W)
            '''
            self.make_attn_map(attn_weights) # (src_attn_map, tar_attn_map)
            self.mask_apply += 1

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if self.is_cross and self.masking and ((self.cur_att_layer+1)//2==30):
            src_attn_map = torch.stack(self.source_attn_map_list, dim=0) # [B, 13, 30, 52]
            tar_attn_map = torch.stack(self.target_attn_map_list, dim=0)
            return hidden_states, (src_attn_map, tar_attn_map)
        else:
            return hidden_states, None


class CleanReferenceAttentionProcessor:
    """
    Clean Reference Self-Attention Processor (PVTT Adaptation).

    Based on RefDrop (NeurIPS 2024) with linear interpolation formula,
    adapted for clean reference images (real product photos).

    Key differences from RefDrop original paper:
    - RefDrop: reference is generated image (participates in denoising)
    - PVTT: reference is real product photo (clean features at t=0)
    - K_ref, V_ref are fixed (extracted once), not dynamic

    Formula: X' = c * Attention(Q, K_ref, V_ref) + (1-c) * Attention(Q, K, V)
    """

    def __init__(
        self,
        ref_bank: Dict[str, Dict[str, torch.Tensor]],
        c: float = 0.2,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
                     Fixed reference features extracted from clean product image (t=0)
            c: Reference guidance coefficient (0.0-1.0)
               0.0 = no reference (pure self-attention)
               0.2 = RefDrop recommended (video generation) ⭐
               1.0 = fully use reference
            layer_indices: Optional list of layer indices to apply reference
                          None = apply to all layers
        """
        self.ref_bank = ref_bank
        self.c = c
        self.layer_indices = layer_indices
        self.cur_att_layer = 0

        print(f"[Clean RefDrop] Initialized with {len(ref_bank)} reference layers")
        print(f"[Clean RefDrop] Guidance coefficient c={c}")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Modified attention forward with RefDrop linear interpolation.

        Formula:
            output = c * Attention(Q, K_ref, V_ref) + (1-c) * Attention(Q, K, V)

        For cross-attention: Normal forward (no modification)
        """
        # Determine if this is cross-attention or self-attention
        is_cross_attention = encoder_hidden_states is not None

        # Handle image conditioning (from I2V models)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization if available
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if available
        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task handling
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            key = torch.cat([key, key_img], dim=2)
            value = torch.cat([value, value_img], dim=2)

        # ===== REFDROP LINEAR INTERPOLATION (only for self-attention) =====
        if not is_cross_attention and self.ref_bank and self.c > 0:
            # 1. Normal self-attention
            attn_self = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            # 2. Find matching layer in ref_bank
            # ref_bank keys are like 'blocks.0.attn1', 'blocks.1.attn1', etc.
            # self.cur_att_layer counts all attention calls (including cross-attention)
            # We need to match self-attention layers only
            layer_key = None
            # Use modulo to cycle through layers for each denoising step
            # 30 blocks × 2 attention (attn1 + attn2) = 60 attention calls per step
            num_layers = len(self.ref_bank)  # 30 layers
            layer_in_step = self.cur_att_layer % (num_layers * 2)  # Reset every step
            block_idx = layer_in_step // 2  # Each block has attn1 + attn2
            candidate_key = f"blocks.{block_idx}.attn1"
            if candidate_key in self.ref_bank:
                layer_key = candidate_key

            # 3. If reference available, compute reference attention
            if layer_key is not None:
                if self.cur_att_layer == 0:  # Print only once per denoising step
                    print(f"[RefDrop] Applying reference attention: {layer_key}")
                K_ref = self.ref_bank[layer_key]["key"]
                V_ref = self.ref_bank[layer_key]["value"]

                # Expand to match batch size
                batch_size = query.shape[0]
                if K_ref.shape[0] == 1 and batch_size > 1:
                    K_ref = K_ref.expand(batch_size, -1, -1)
                    V_ref = V_ref.expand(batch_size, -1, -1)

                # Reshape K_ref, V_ref to match multi-head format
                head_dim = query.shape[-1]
                K_ref = K_ref.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
                V_ref = V_ref.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)

                # Compute reference attention
                attn_ref = F.scaled_dot_product_attention(
                    query, K_ref, V_ref, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

                # 4. RefDrop linear interpolation ⭐ CORE FORMULA
                hidden_states = self.c * attn_ref + (1 - self.c) * attn_self
            else:
                # No reference for this layer, use normal attention
                hidden_states = attn_self
        else:
            # Cross-attention or no reference: normal forward
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Increment layer counter
        self.cur_att_layer += 1

        return hidden_states, None  # Return (hidden_states, attn_map) like MasaCtrlProcessor


class NoisyReferenceAttentionProcessor:
    """
    Noisy Reference Self-Attention Processor (PVTT Improved).

    Unlike CleanReferenceAttentionProcessor which uses fixed features from t=0,
    this processor pre-computes features for all timesteps, then uses the
    matching features during denoising.

    Formula: X' = c * Attention(Q, K_ref, V_ref) + (1-c) * Attention(Q, K, V)
             where K_ref, V_ref are from noisy reference at current timestep
    """

    def __init__(
        self,
        ref_latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        transformer,
        scheduler,
        timesteps: torch.Tensor,  # All timesteps used in denoising
        c: float = 0.2,
    ):
        """
        Args:
            ref_latent: Clean reference latent [1, 16, 1, H', W']
            prompt_embeds: Text embeddings [1, seq_len, dim]
            transformer: Transformer model (for feature extraction)
            scheduler: Noise scheduler (for adding noise)
            timesteps: All timesteps that will be used in denoising
            c: Reference guidance coefficient (0.0-1.0)
        """
        self.ref_latent = ref_latent
        self.prompt_embeds = prompt_embeds
        self.transformer = transformer
        self.scheduler = scheduler
        self.c = c

        self.cur_att_layer = 0
        self.current_step_idx = 0

        # Pre-compute features for all timesteps
        print(f"[Noisy RefDrop] Pre-computing features for {len(timesteps)} timesteps...")
        self.all_ref_banks = self._precompute_all_features(timesteps)
        print(f"[Noisy RefDrop] Initialized with c={c}, {len(self.all_ref_banks)} steps cached")

        # Register a simple hook to track current timestep
        self._register_timestep_hook()

    def _register_timestep_hook(self):
        """Register hook to track current timestep during denoising."""
        def hook_fn(module, args, kwargs):
            timestep = kwargs.get('timestep', None)
            if timestep is None and len(args) > 1:
                timestep = args[1]
            if timestep is not None:
                self.set_step(timestep)

        self._hook = self.transformer.register_forward_pre_hook(hook_fn, with_kwargs=True)

    def _precompute_all_features(self, timesteps):
        """Pre-compute reference features for all timesteps (stored on CPU to save GPU memory)."""
        from .reference_utils import extract_noisy_reference_features

        all_banks = {}
        for i, t in enumerate(timesteps):
            t_val = t.item() if isinstance(t, torch.Tensor) else t

            ref_bank = extract_noisy_reference_features(
                ref_latent=self.ref_latent,
                prompt_embeds=self.prompt_embeds,
                transformer=self.transformer,
                scheduler=self.scheduler,
                timestep=t,
                device=self.ref_latent.device,
            )

            # Move to CPU to save GPU memory
            ref_bank_cpu = {}
            for layer_name, layer_data in ref_bank.items():
                ref_bank_cpu[layer_name] = {
                    "key": layer_data["key"].cpu(),
                    "value": layer_data["value"].cpu(),
                }
            all_banks[t_val] = ref_bank_cpu

            # Clear GPU cache periodically
            if i % 5 == 0:
                torch.cuda.empty_cache()

            if i % 10 == 0 or i == len(timesteps) - 1:
                print(f"[Noisy RefDrop] Pre-computed {i+1}/{len(timesteps)} (t={int(t_val)})")

        return all_banks

    def set_step(self, timestep):
        """Set current timestep for feature lookup."""
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t_val = timestep
        self.current_timestep = t_val
        self.cur_att_layer = 0

    def cleanup(self):
        """Cleanup resources."""
        self.all_ref_banks = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Modified attention forward with Noisy RefDrop.
        """
        is_cross_attention = encoder_hidden_states is not None

        # Handle image conditioning (from I2V models)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V handling
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = torch.cat([key, key_img], dim=2)
            value = torch.cat([value, value_img], dim=2)

        # ===== NOISY REFDROP (only for self-attention) =====
        # Get cached features for current timestep
        cached_ref_bank = None
        if hasattr(self, 'current_timestep') and self.current_timestep is not None:
            cached_ref_bank = self.all_ref_banks.get(self.current_timestep, None)

        if not is_cross_attention and cached_ref_bank and self.c > 0:
            # Normal self-attention
            attn_self = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            # Find matching layer
            layer_key = None
            num_layers = len(cached_ref_bank)
            if num_layers > 0:
                layer_in_step = self.cur_att_layer % (num_layers * 2)
                block_idx = layer_in_step // 2
                candidate_key = f"blocks.{block_idx}.attn1"
                if candidate_key in cached_ref_bank:
                    layer_key = candidate_key

            if layer_key is not None:
                # Move from CPU to GPU and match dtype
                K_ref = cached_ref_bank[layer_key]["key"].to(query.device, dtype=query.dtype)
                V_ref = cached_ref_bank[layer_key]["value"].to(query.device, dtype=query.dtype)

                batch_size = query.shape[0]
                if K_ref.shape[0] == 1 and batch_size > 1:
                    K_ref = K_ref.expand(batch_size, -1, -1)
                    V_ref = V_ref.expand(batch_size, -1, -1)

                head_dim = query.shape[-1]
                K_ref = K_ref.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
                V_ref = V_ref.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)

                attn_ref = F.scaled_dot_product_attention(
                    query, K_ref, V_ref, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

                # RefDrop linear interpolation
                hidden_states = self.c * attn_ref + (1 - self.c) * attn_self
            else:
                hidden_states = attn_self
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        self.cur_att_layer += 1

        return hidden_states, None
