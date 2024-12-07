import torch.nn.functional as F

class ControlNetApplyAdvancedMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "control_net": ("CONTROL_NET", ),
            "image": ("IMAGE", ),
            "mask": ("MASK", ),  # ComfyUI MASK type
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "vae": ("VAE", ),
                }
            }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"

    def process_mask(self, mask):        
        # Ensure mask is in [1, H, W] format
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) != 3:
            raise ValueError(f"Expected mask shape [1, H, W], got: {mask.shape}")
        
        # Downsample mask to latent dimensions (H/16, W/16)
        _, h, w = mask.shape
        target_size = (h // 16, w // 16)
        
        mask = F.interpolate(
            mask.unsqueeze(0),  # Add batch dim [1, 1, H, W]
            size=target_size,
            mode='nearest'
        ).squeeze(0)  # Back to [1, H, W]
        
        return mask

    def apply_mask_to_control(self, control_signal, mask):
        
        # Get dimensions
        batch, flattened_dim, feature_dim = control_signal.shape
        _, mask_h, mask_w = mask.shape
        
        # Verify dimensions match
        if mask_h * mask_w != flattened_dim:
            raise ValueError(f"Mask dimensions ({mask_h}x{mask_w}={mask_h*mask_w}) "
                            f"don't match flattened control signal dimension ({flattened_dim})")
        
        # Flatten and expand mask to match control signal
        mask_flattened = mask.reshape(batch, flattened_dim, 1).expand(-1, -1, feature_dim)
        
        # Apply mask
        masked_signal = control_signal * mask_flattened.to(control_signal.device)
        
        return masked_signal

    def apply_controlnet(self, positive, negative, control_net, image, mask, strength, start_percent, end_percent, vae=None):
        if strength == 0:
            return (positive, negative)

        feature_mask = self.process_mask(mask)
        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # Create new control net instance
                    c_net = control_net.copy()

                    # Store original method
                    original_get_control = c_net.get_control

                    def create_masked_get_control(original_fn, feature_mask):
                        def masked_get_control(*args, **kwargs):
                            control_output = original_fn(*args, **kwargs)

                            if isinstance(control_output, dict):
                                for key in control_output:
                                    if control_output[key] is not None:
                                        for i in range(len(control_output[key])):
                                            if control_output[key][i] is not None:
                                                control_output[key][i] = self.apply_mask_to_control(
                                                        control_output[key][i], 
                                                        feature_mask
                                                        )
                            return control_output
                        return masked_get_control

                    # Set the masked version of get_control
                    c_net.get_control = create_masked_get_control(original_get_control, feature_mask)

                    # Set the condition hint
                    c_net = c_net.set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)

        return (out[0], out[1])



NODE_CLASS_MAPPINGS = {
    "ControlNetApplyAdvancedMasked": ControlNetApplyAdvancedMasked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetApplyAdvancedMasked": "Apply ControlNet (Advanced+Mask)",
    
}