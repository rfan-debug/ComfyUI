import comfy.model_management as mm
import folder_paths
import logging
import comfy
import torch
from .controlnet.controlnet_instantx import InstantXControlNetFlux
from .controlnet.controlnet_instantx_format2 import InstantXControlNetFluxFormat2
from comfy.controlnet import ControlNet, controlnet_load_state_dict
from nodes import ControlNetApplyAdvanced

def load_controlnet_flux_instantx(sd, controlnet_class, weight_dtype):
    keys_to_keep = [
            "controlnet_",
            "single_transformer_blocks",
            "transformer_blocks"
            ]
    preserved_keys = {k: v.cpu() for k, v in sd.items() if any(k.startswith(key) for key in keys_to_keep)}

    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")

    keys_to_discard = [
            "double_blocks", 
            "single_blocks"
            ]
    new_sd = {k: v for k, v in new_sd.items() if not any(k.startswith(discard_key) for discard_key in keys_to_discard)}
    new_sd.update(preserved_keys)

    config = {
            "image_model": "flux",
            "axes_dim": [16, 56, 56],
            "in_channels": 16,
            "depth": 5,
            "depth_single_blocks": 10,
            "context_in_dim": 4096,
            "num_heads": 24,
            "guidance_embed": True,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "theta": 10000,
            "qkv_bias": True,
            "vec_in_dim": 768
            }

    device=mm.get_torch_device()

    if weight_dtype == "fp8_e4m3fn":
        dtype=torch.float8_e4m3fn
        operations = comfy.ops.manual_cast
    elif weight_dtype == "fp8_e5m2":
        dtype=torch.float8_e5m2
        operations = comfy.ops.manual_cast
    else:
        dtype=torch.bfloat16
        operations = comfy.ops.disable_weight_init

    control_model = controlnet_class(operations=operations, device=device, dtype=dtype, **config)
    control_model = controlnet_load_state_dict(control_model, new_sd)
    extra_conds = ['y', 'guidance', 'control_type']
    latent_format = comfy.latent_formats.SD3()
    # TODO check manual cast dtype
    control = ControlNet(control_model, compression_ratio=1, load_device=device, manual_cast_dtype=torch.bfloat16, extra_conds=extra_conds, latent_format=latent_format)
    return control

def load_controlnet(full_path, weight_dtype):
    controlnet_data = comfy.utils.load_torch_file(full_path, safe_load=True)
    if "controlnet_mode_embedder.fc.weight" in controlnet_data:
        return load_controlnet_flux_instantx(controlnet_data, InstantXControlNetFlux, weight_dtype)
    if "controlnet_mode_embedder.weight" in controlnet_data:
        return load_controlnet_flux_instantx(controlnet_data, InstantXControlNetFluxFormat2, weight_dtype)
    assert False, f"Only InstantX union controlnet supported. Could not find key 'controlnet_mode_embedder.fc.weight' in {full_path}"

INSTANTX_UNION_CONTROLNET_TYPES = {
        "canny": 0,
        "tile": 1,
        "depth": 2,
        "blur": 3,
        "pose": 4,
        "gray": 5,
        "lq": 6
        }

class InstantXFluxUnionControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                    "type": (list(INSTANTX_UNION_CONTROLNET_TYPES.keys()),),
                    #"weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],)
                    }
                }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, type, weight_dtype="default"):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, weight_dtype)

        type_number = INSTANTX_UNION_CONTROLNET_TYPES.get(type, -1)
        controlnet.set_extra_arg("control_type", type_number)

        return (controlnet,)

import torch
import torch.nn.functional as F

class ControlNetApplyAdvancedMasked():

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "control_net": ("CONTROL_NET",),
            "vae": ("VAE", ),
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {"attention_mask": ("MASK",)}
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"

    def apply_controlnet(self, positive, negative, control_net, vae, image, strength, start_percent, end_percent, attention_mask=None):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        print(f"image.shape {image.shape}")
        print(f"attention_mask.shape {attention_mask.shape}")

        # Process attention attention_mask if provided
        if attention_mask is not None:
            h, w = attention_mask.shape[-2:]
            assert(h%16 ==0)
            assert(w%16 == 0)
            assert(attention_mask.shape[1:3] == image.shape[1:3])


        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae, )
                    c_net.set_previous_controlnet(prev_cnet)
                    if attention_mask is not None:
                        # Store the attention attention_mask in the controlnet's extra_args

                        print(f"in apply attention_mask: {attention_mask.shape}")
                        c_net.set_extra_arg("attention_mask", attention_mask.cuda())

                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])