import gradio
import torch
import einops

import modules
from modules.processing import StableDiffusionProcessing
import modules.scripts
from modules import sd_samplers
from backend import memory_management
from backend.args import args, dynamic_args

from scripts.classic_engine import ClassicTextProcessingEngineTopKEmphasis
import scripts.prompt_parser as prompt_parser

filtering_locations = ["Text Processing", "After K", "After QK", "After QKV"]
zero_point = ["Zero", "Mean of Prompt", "Median of Prompt", "Mean of Padding", "Median of padding"]

class TopKEmphasis(modules.scripts.Script):
    """
    Applies emphasis at following locations:
        "c": After conditioning has been constructed.
        "b": Before every k in cross attention.
        "k": After every k in cross attention.
        "q": After every q * v in cross attention.
        "s": After every softmax in cross attention.
        "v": After every v in cross attention.
    """

    model_type = None
    text_processing_engine_original = None
    text_processing_engine_l_original = None
    text_processing_engine_g_original = None
    positive_multiplier = None
    negative_multiplier = None

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Top K Emphasis"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        active = gradio.Checkbox(value=False, label="Enable Top K Emphasis")
        return [active]
    
    def before_process(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        if not active: return
        print("Loading Top K Emphasis.")
        if hasattr(p.sd_model, "text_processing_engine"):
            TopKEmphasis.model_type = "SD1.5"
            TopKEmphasis.text_processing_engine_original = p.sd_model.text_processing_engine
            if not isinstance(p.sd_model.text_processing_engine, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine.tokenizer,
                    embedding_dir=dynamic_args['embedding_dir'],
                    embedding_key='clip_l',
                    embedding_expected_shape=768,
                    emphasis_name=dynamic_args['emphasis_name'],
                    text_projection=False,
                    minimal_clip_skip=1,
                    clip_skip=1,
                    return_pooled=False,
                    final_layer_norm=True,
                )
            hook_forwards(p.sd_model.forge_objects.unet.model, False)
        elif hasattr(p.sd_model, "text_processing_engine_l") and hasattr(p.sd_model, "text_processing_engine_g"):
            TopKEmphasis.model_type = "SDXL"
            TopKEmphasis.text_processing_engine_l_original = p.sd_model.text_processing_engine_l
            if not isinstance(p.sd_model.text_processing_engine_l, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine_l = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine_l.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine_l.tokenizer,
                    embedding_dir=dynamic_args['embedding_dir'],
                    embedding_key='clip_l',
                    embedding_expected_shape=2048,
                    emphasis_name=dynamic_args['emphasis_name'],
                    text_projection=False,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=False,
                    final_layer_norm=False,
                )
            TopKEmphasis.text_processing_engine_g_original = p.sd_model.text_processing_engine_g
            if not isinstance(p.sd_model.text_processing_engine_g, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine_g = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine_g.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine_g.tokenizer,
                    embedding_dir=dynamic_args['embedding_dir'],
                    embedding_key='clip_g',
                    embedding_expected_shape=2048,
                    emphasis_name=dynamic_args['emphasis_name'],
                    text_projection=True,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=True,
                    final_layer_norm=False,
                )
            hook_forwards(p.sd_model.forge_objects.unet.model, False)
        else:
            raise Exception("Unsupported model type.")

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        p.setup_conds = lambda: None
        p.c, TopKEmphasis.positive_multiplier, p.uc, TopKEmphasis.negative_multiplier = setup_conds(p)

    def postprocess(self, p: StableDiffusionProcessing, processed, active, *args):
        if not active: return
        match TopKEmphasis.model_type:
            case "SD1.5":
                if TopKEmphasis.text_processing_engine_original is not None:
                    p.sd_model.text_processing_engine = TopKEmphasis.text_processing_engine_original
                hook_forwards(p.sd_model.forge_objects.unet.model, True)
            case "SDXL":
                if TopKEmphasis.text_processing_engine_l_original is not None:
                    p.sd_model.text_processing_engine_l = TopKEmphasis.text_processing_engine_l_original
                if TopKEmphasis.text_processing_engine_g_original is not None:
                    p.sd_model.text_processing_engine_g = TopKEmphasis.text_processing_engine_g_original
                hook_forwards(p.sd_model.forge_objects.unet.model, True)
        print("Unloading Top K Emphasis.")

def setup_conds(self: StableDiffusionProcessing):
    prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height, distilled_cfg_scale=self.distilled_cfg_scale)
    negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True, distilled_cfg_scale=self.distilled_cfg_scale)

    sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
    total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
    self.step_multiplier = total_steps // self.steps
    self.firstpass_steps = total_steps

    if self.cfg_scale == 1:
        uc = None
        print('Skipping unconditional conditioning when CFG = 1. Negative Prompts are ignored.')
    else:
        uc, um = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)

    c, m = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)
    return c, m, uc, um

def hook_forward(top_k_emphasis: TopKEmphasis, module):
    FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()

    def get_attn_precision(attn_precision=torch.float32):
        if args.disable_attention_upcast:
            return None
        if FORCE_UPCAST_ATTENTION_DTYPE is not None:
            return FORCE_UPCAST_ATTENTION_DTYPE
        return attn_precision

    def exists(val):
        return val is not None

    def forward(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
        attn_precision = get_attn_precision(attn_precision)

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        scale = dim_head ** -0.5

        h = heads
        if skip_reshape:
            q, k, v = map(
                lambda t: t.reshape(b * heads, -1, dim_head),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, -1, heads, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * heads, -1, dim_head)
                .contiguous(),
                (q, k, v),
            )

        if attn_precision == torch.float32:
            sim = torch.einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

        del q, k

        if exists(mask):
            if mask.dtype == torch.bool:
                mask = einops.rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            else:
                if len(mask.shape) == 2:
                    bs = 1
                else:
                    bs = mask.shape[0]
                mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
                sim.add_(mask)

        sim = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out
    return forward

def hook_forwards(top_k_emphasis: TopKEmphasis, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(top_k_emphasis, module)
            if remove:
                del module.forward