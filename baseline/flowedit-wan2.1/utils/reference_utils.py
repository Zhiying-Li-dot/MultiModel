"""
Reference Self-Attention utilities for PVTT baseline.

Based on RefDrop (NeurIPS 2024) method, adapted for reference images.

Two modes:
1. Clean RefDrop (original): Extract features at t=0, reuse for all steps
   - Problem: Feature distribution mismatch at high noise levels

2. Noisy RefDrop (new): Add noise to reference matching current timestep
   - Each step: noisy_ref = add_noise(clean_ref, t)
   - Extract K_ref, V_ref from noisy_ref
   - Features match the noise level of current denoising step
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from PIL import Image
import torchvision.transforms.functional as TF


def preprocess_image(image: Image.Image, target_size: tuple = None) -> torch.Tensor:
    """
    Preprocess PIL image to tensor for VAE encoding.

    Args:
        image: PIL Image
        target_size: (height, width) tuple, if None uses image size

    Returns:
        tensor: [3, H, W] normalized to [-1, 1]
    """
    if target_size is not None:
        h, w = target_size
        # Resize with maintaining aspect ratio and center crop
        scale = max(w / image.width, h / image.height)
        new_w = round(image.width * scale)
        new_h = round(image.height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Center crop
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        image = image.crop((left, top, left + w, top + h))

    # Convert to tensor and normalize to [-1, 1]
    tensor = TF.to_tensor(image)
    tensor = tensor * 2.0 - 1.0

    return tensor


def extract_clean_reference_features(
    product_image: Image.Image,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    ref_prompt: str,
    target_size: tuple = (480, 832),
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract clean reference features from product image (PVTT Adaptation).

    ⚠️ PVTT 适配关键：
    - 在 timestep=0 提取 features（clean state，无噪声）
    - 所有 denoising step 共用这些固定的 K_ref, V_ref
    - 不随生成过程的 timestep 变化
    - 类似 ControlNet 的做法

    This function:
    1. Encodes clean product image to VAE latent
    2. Forward through transformer at t=0 with hooks to extract K, V
    3. Returns a bank of fixed reference features for all denoising steps

    Args:
        product_image: Real product image (PIL Image, clean)
        vae: VAE model for encoding
        transformer: Transformer model (WanTransformer3DModel)
        text_encoder: T5 text encoder
        tokenizer: T5 tokenizer
        ref_prompt: Text prompt for reference (typically target_prompt)
        target_size: (height, width) for resizing image
        device: Device to run on

    Returns:
        ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
                 Each K_ref, V_ref has shape [1, seq_len, dim]
                 Fixed features extracted at t=0
    """
    print("[Clean RefDrop] Extracting features from product image...")

    # Step 1: Preprocess and encode image to latent
    ref_tensor = preprocess_image(product_image, target_size)
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, H, W]

    with torch.no_grad():
        # Encode to latent space (no noise!)
        z_ref = vae.encode(ref_tensor).latent_dist.sample()  # [1, 16, 1, H', W']
        print(f"[Clean RefDrop] Latent shape: {z_ref.shape}")

        # Step 2: Encode reference prompt
        prompt_embeds = text_encoder(
            tokenizer(
                ref_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
        )[0]

        # Step 3: Prepare feature extraction with hooks
        ref_bank = {}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, args, kwargs, output):
                # Extract features from self-attention
                # For WAN transformer, hidden_states is the input to attention
                # NOTE: with_kwargs=True changes signature to (module, args, kwargs, output)

                # Try to get hidden_states from either args or kwargs
                hidden_states = None
                encoder_hidden_states = None

                if len(args) > 0:
                    hidden_states = args[0]
                    if len(args) > 1:
                        encoder_hidden_states = args[1]
                elif kwargs:
                    hidden_states = kwargs.get('hidden_states', None)
                    encoder_hidden_states = kwargs.get('encoder_hidden_states', None)

                if hidden_states is not None:
                    # Check if this is self-attention (no encoder_hidden_states)
                    is_self_attn = encoder_hidden_states is None

                    if is_self_attn:
                        # This is self-attention
                        try:
                            K_ref = module.to_k(hidden_states)
                            V_ref = module.to_v(hidden_states)

                            ref_bank[layer_name] = {
                                "key": K_ref.detach().clone(),
                                "value": V_ref.detach().clone()
                            }
                        except AttributeError:
                            # Module doesn't have to_k/to_v, skip
                            pass

            return hook_fn

        # Step 4: Register hooks to all attention modules
        for name, module in transformer.named_modules():
            # Look for attention modules
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # Use register_forward_hook with with_kwargs=True to capture keyword arguments
                hook = module.register_forward_hook(make_hook(name), with_kwargs=True)
                hooks.append(hook)

        print(f"[Clean RefDrop] Registered {len(hooks)} hooks")

        # Step 5: Forward pass at t=0 (clean state) ⭐ 关键
        t = torch.zeros(1, device=device)  # t=0 for clean features

        # Convert to transformer's dtype (float16 for T2V-1.3B)
        z_ref = z_ref.to(transformer.dtype)
        prompt_embeds = prompt_embeds.to(transformer.dtype)

        # Temporarily register basic attention processor for feature extraction
        # (default processor may not return correct tuple format)
        from .wan_attention import WanAttnProcessor2_0, register_attention_processor
        register_attention_processor(transformer, processor_type="WanAttnProcessor2_0")

        # Forward through transformer
        try:
            _ = transformer(
                z_ref,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
                return_dict=False
            )
        except Exception as e:
            print(f"[Clean RefDrop] Warning: Forward pass encountered error: {e}")
            # Continue anyway, we may have extracted some features

        # Step 6: Remove hooks
        for hook in hooks:
            hook.remove()

        print(f"[Clean RefDrop] Extracted features from {len(ref_bank)} layers (t=0)")

        # Print sample layer info
        if ref_bank:
            sample_name = list(ref_bank.keys())[0]
            sample_k = ref_bank[sample_name]["key"]
            print(f"[Clean RefDrop] Sample layer '{sample_name}' K shape: {sample_k.shape}")

    return ref_bank


def prepare_reference_latent(
    product_image: Image.Image,
    vae,
    text_encoder,
    tokenizer,
    ref_prompt: str,
    target_size: tuple = (480, 832),
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare reference latent and prompt embeddings for Noisy RefDrop.

    Unlike extract_clean_reference_features, this only encodes the image
    to latent space. Feature extraction happens dynamically at each timestep.

    Args:
        product_image: Real product image (PIL Image)
        vae: VAE model for encoding
        text_encoder: T5 text encoder
        tokenizer: T5 tokenizer
        ref_prompt: Text prompt for reference
        target_size: (height, width) for resizing image
        device: Device to run on

    Returns:
        dict with:
            - ref_latent: [1, 16, 1, H', W'] clean latent
            - prompt_embeds: [1, seq_len, dim] text embeddings
    """
    print("[Noisy RefDrop] Preparing reference latent...")

    # Preprocess and encode image to latent
    ref_tensor = preprocess_image(product_image, target_size)
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, H, W]

    with torch.no_grad():
        # Encode to latent space
        ref_latent = vae.encode(ref_tensor).latent_dist.sample()
        print(f"[Noisy RefDrop] Reference latent shape: {ref_latent.shape}")

        # Encode reference prompt
        prompt_embeds = text_encoder(
            tokenizer(
                ref_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
        )[0]

    return {
        "ref_latent": ref_latent,
        "prompt_embeds": prompt_embeds,
    }


def extract_noisy_reference_features(
    ref_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    transformer,
    scheduler,
    timestep: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract reference features from noisy latent at given timestep.

    This is called once per denoising step to get features matching
    the current noise level.

    Args:
        ref_latent: Clean reference latent [1, 16, 1, H', W']
        prompt_embeds: Text embeddings [1, seq_len, dim]
        transformer: Transformer model
        scheduler: Noise scheduler (for adding noise)
        timestep: Current denoising timestep
        device: Device

    Returns:
        ref_bank: {layer_name: {"key": K_ref, "value": V_ref}}
    """
    with torch.no_grad():
        # Add noise to reference latent matching current timestep
        noise = torch.randn_like(ref_latent)

        # Ensure timestep is 1-d tensor (scheduler expects iterable)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        # Add noise using scheduler
        noisy_ref = scheduler.add_noise(ref_latent, noise, timestep)

        # Prepare feature extraction with hooks
        ref_bank = {}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, args, kwargs, output):
                hidden_states = None
                encoder_hidden_states = None

                if len(args) > 0:
                    hidden_states = args[0]
                    if len(args) > 1:
                        encoder_hidden_states = args[1]
                elif kwargs:
                    hidden_states = kwargs.get('hidden_states', None)
                    encoder_hidden_states = kwargs.get('encoder_hidden_states', None)

                if hidden_states is not None:
                    is_self_attn = encoder_hidden_states is None
                    if is_self_attn:
                        try:
                            K_ref = module.to_k(hidden_states)
                            V_ref = module.to_v(hidden_states)
                            ref_bank[layer_name] = {
                                "key": K_ref.detach().clone(),
                                "value": V_ref.detach().clone()
                            }
                        except AttributeError:
                            pass
            return hook_fn

        # Register hooks
        for name, module in transformer.named_modules():
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                hook = module.register_forward_hook(make_hook(name), with_kwargs=True)
                hooks.append(hook)

        # Convert to transformer's dtype
        noisy_ref = noisy_ref.to(transformer.dtype)
        prompt_embeds_t = prompt_embeds.to(transformer.dtype)

        # Temporarily use basic attention processor
        from .wan_attention import register_attention_processor
        register_attention_processor(transformer, processor_type="WanAttnProcessor2_0")

        # Forward through transformer at current timestep
        try:
            _ = transformer(
                noisy_ref,
                encoder_hidden_states=prompt_embeds_t,
                timestep=timestep.to(transformer.dtype),
                return_dict=False
            )
        except Exception as e:
            print(f"[Noisy RefDrop] Warning: Forward pass error: {e}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

    return ref_bank
