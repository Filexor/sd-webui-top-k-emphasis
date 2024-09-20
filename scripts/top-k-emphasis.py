import gradio
import torch
import einops

import modules
from modules.devices import device
from modules.processing import StableDiffusionProcessing
import modules.scripts
from modules import sd_samplers
from backend import memory_management
from backend.args import args, dynamic_args
from backend.attention import attention_function

from scripts.classic_engine import ClassicTextProcessingEngineTopKEmphasis
from scripts.parsing import EmphasisPair
import scripts.prompt_parser as prompt_parser

filtering_locations = ["Text Processing", "After K", "After QK", "After QKV"]
zero_point = ["Zero", "Mean of Prompt", "Median of Prompt", "Mean of Padding", "Median of padding"]

class TopKEmphasis(modules.scripts.Script):
    """
    Applies emphasis at following locations:
        "c": After conditioning has been constructed.
        "k": After every k in cross attention.
        "v": After every v in cross attention.
    Following can used with "Enbale Extra Mode":
        "q": After every dot(q, k) in cross attention.
    """

    extra_mode = False
    model_type = None
    text_processing_engine_original = None
    text_processing_engine_l_original = None
    text_processing_engine_g_original = None
    get_learned_conditioning_sdxl_original = None
    positive_multiplier = None
    negative_multiplier = None
    current_step_q_mul = None
    current_step_q_thres = None
    current_step_k_mul = None
    current_step_k_thres = None
    current_step_v_mul = None
    current_step_v_thres = None

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Top K Emphasis"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gradio.Accordion("Top K Emphasis", open=False):
            with gradio.Row():
                active = gradio.Checkbox(value=False, label="Enable Top K Emphasis")
                extra_mode = gradio.Checkbox(value=False, label="Enbale Extra Mode")
        return [active, extra_mode]
    
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
            hook_forwards(TopKEmphasis, p.sd_model.forge_objects.unet.model, False)
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
            TopKEmphasis.get_learned_conditioning_sdxl_original = p.sd_model.get_learned_conditioning
            p.sd_model.get_learned_conditioning = get_learned_conditioning_sdxl
            hook_forwards(TopKEmphasis, p.sd_model.forge_objects.unet.model, False)
        else:
            raise Exception("Unsupported model type.")

    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        if not active: return
        p.setup_conds = lambda: None
        p.cached_c = [None, None, None]
        p.cached_uc = [None, None, None]
        p.c, TopKEmphasis.positive_multiplier, p.uc, TopKEmphasis.negative_multiplier = setup_conds(p)

    def process_before_every_sampling(self, p: StableDiffusionProcessing, active, extra_mode, *args, **kwargs):
        if not active: return
        TopKEmphasis.extra_mode = extra_mode
        pm = prompt_parser.reconstruct_multi_multiplier_batch(TopKEmphasis.positive_multiplier, p.steps)
        nm = prompt_parser.reconstruct_multiplier_batch(TopKEmphasis.negative_multiplier, p.steps)
        d = len(pm) - len(nm)
        if d > 0:
            nm += [EmphasisPair()] * d
        elif d < 0:
            pm += [EmphasisPair()] * -d
        pmq, ptq = to_structure_of_tensor(pm, "q")
        nmq, ntq = to_structure_of_tensor(nm, "q")
        TopKEmphasis.current_step_q_mul = torch.stack((pmq, nmq), dim=1)
        TopKEmphasis.current_step_q_thres = torch.stack((ptq, ntq), dim=1)
        pmk, ptk = to_structure_of_tensor(pm, "k")
        nmk, ntk = to_structure_of_tensor(nm, "k")
        TopKEmphasis.current_step_k_mul = torch.stack((pmk, nmk), dim=1)
        TopKEmphasis.current_step_k_thres = torch.stack((ptk, ntk), dim=1)
        pmv, ptv = to_structure_of_tensor(pm, "v")
        nmv, ntv = to_structure_of_tensor(nm, "v")
        TopKEmphasis.current_step_v_mul = torch.stack((pmv, nmv), dim=1)
        TopKEmphasis.current_step_v_thres = torch.stack((ptv, ntv), dim=1)

    def postprocess(self, p: StableDiffusionProcessing, processed, active, *args):
        if not active: return
        match TopKEmphasis.model_type:
            case "SD1.5":
                if TopKEmphasis.text_processing_engine_original is not None:
                    p.sd_model.text_processing_engine = TopKEmphasis.text_processing_engine_original
                hook_forwards(TopKEmphasis, p.sd_model.forge_objects.unet.model, True)
            case "SDXL":
                if TopKEmphasis.text_processing_engine_l_original is not None:
                    p.sd_model.text_processing_engine_l = TopKEmphasis.text_processing_engine_l_original
                if TopKEmphasis.text_processing_engine_g_original is not None:
                    p.sd_model.text_processing_engine_g = TopKEmphasis.text_processing_engine_g_original
                if TopKEmphasis.get_learned_conditioning_sdxl_original is not None:
                    p.sd_model.get_learned_conditioning = TopKEmphasis.get_learned_conditioning_sdxl_original
                hook_forwards(TopKEmphasis, p.sd_model.forge_objects.unet.model, True)
        print("Unloading Top K Emphasis.")

@torch.inference_mode()
def get_learned_conditioning_sdxl(self, prompt: list[str]):
    memory_management.load_model_gpu(self.forge_objects.clip.patcher)

    cond_l, multiplier_l = self.text_processing_engine_l(prompt)
    cond_g, clip_pooled, multiplier_g = self.text_processing_engine_g(prompt)

    width = getattr(prompt, 'width', 1024) or 1024
    height = getattr(prompt, 'height', 1024) or 1024
    is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

    crop_w = 0
    crop_h = 0
    target_width = width
    target_height = height

    out = [
        self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
        self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
        self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))
    ]

    flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1).to(clip_pooled)

    force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)

    if force_zero_negative_prompt:
        clip_pooled = torch.zeros_like(clip_pooled)
        cond_l = torch.zeros_like(cond_l)
        cond_g = torch.zeros_like(cond_g)

    cond = dict(
        crossattn=torch.cat([cond_l, cond_g], dim=2),
        vector=torch.cat([clip_pooled, flat], dim=1),
    )

    return cond, (multiplier_l, multiplier_g)

def setup_conds(self: StableDiffusionProcessing):
    prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height, distilled_cfg_scale=self.distilled_cfg_scale)
    negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True, distilled_cfg_scale=self.distilled_cfg_scale)

    sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
    total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
    self.step_multiplier = total_steps // self.steps
    self.firstpass_steps = total_steps

    if self.cfg_scale == 1:
        uc, um = None
        print('Skipping unconditional conditioning when CFG = 1. Negative Prompts are ignored.')
    else:
        uc, um = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)

    c, m = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)
    return c, m, uc, um

def to_structure_of_tensor(input: list[EmphasisPair], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    count = 1
    weights = []
    thresholds = []
    for i in input:
        weight = []
        threshold = []
        count_tokenwise = 0
        for j in i.multipliers:
            if j.key == key:
                count_tokenwise += 1
                weight.append(j.weight)
                threshold.append(j.threshold)
        count = max(count, count_tokenwise)
        weights.append(weight)
        thresholds.append(threshold)
    for i in weights:
        i += [1.0] * (count - len(i))
    for i in thresholds:
        i += [0.0] * (count - len(i))
    weight_t = torch.asarray(weights)
    weight_t = einops.rearrange(weight_t, "a b -> b a")
    threshold_t = torch.asarray(thresholds)
    threshold_t = einops.rearrange(threshold_t, "a b -> b a")
    return weight_t, threshold_t

def hook_forward(top_k_emphasis: TopKEmphasis, self):
    FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()

    def get_attn_precision(attn_precision=torch.float32):
        if args.disable_attention_upcast:
            return None
        if FORCE_UPCAST_ATTENTION_DTYPE is not None:
            return FORCE_UPCAST_ATTENTION_DTYPE
        return attn_precision

    def exists(val):
        return val is not None
    
    def default(val, d):
        if exists(val):
            return val
        return d
    
    def apply_top_k_emphasis(z: torch.Tensor, key: str):
        if key == "k":
            multiplier = TopKEmphasis.current_step_k_mul
            threshold = TopKEmphasis.current_step_k_thres   # [depth, c/uc, token]
        elif key == "v":
            multiplier = TopKEmphasis.current_step_v_mul
            threshold = TopKEmphasis.current_step_v_thres   # [depth, c/uc, token]
        token_count = multiplier.shape[-1]
        threshold = torch.where(threshold == 0.0, z.shape[1], threshold)
        threshold = torch.where(threshold < 1.0, threshold * z.shape[1], threshold)
        threshold = threshold - 1
        threshold = threshold.to(dtype=torch.int32)
        threshold = einops.rearrange(threshold, "a b e -> a (b e)").to(device)
        multiplier = einops.rearrange(multiplier, "a b e -> a (b e)").to(device)
        z_dec = einops.rearrange(z, "a c d -> d (a c)").sort(dim=0).values
        z = einops.rearrange(z, "a c d -> d (a c)")
        for i in range(threshold.shape[0]):
            selected_z_dec = z_dec.index_select(dim=0, index=threshold[i, :])[0, :]
            expanded_z_dec = selected_z_dec.unsqueeze(0).expand(z.shape[0], -1)
            expanded_multiplier = multiplier[i, :].unsqueeze(0).expand(z.shape[0], -1)
            z *= torch.where(z >= expanded_z_dec, expanded_multiplier, 1.0)
        z = einops.rearrange(z, "d (a c) -> a c d", c=token_count)

    def apply_top_k_emphasis_q(z: torch.Tensor, heads):
        multiplier = TopKEmphasis.current_step_q_mul
        threshold = TopKEmphasis.current_step_q_thres   # [depth, c/uc, token]
        token_count = multiplier.shape[-1]
        threshold = torch.where(threshold == 0.0, z.shape[1], threshold)
        threshold = torch.where(threshold < 1.0, threshold * z.shape[1], threshold)
        threshold = threshold - 1
        threshold = threshold.to(dtype=torch.int32)
        threshold = threshold[:, :, :].repeat_interleave(heads, dim=2).to(device)
        multiplier = multiplier[:, :, :].repeat_interleave(heads, dim=2).to(device)
        threshold = einops.rearrange(threshold, "a b e -> a (b e)")
        multiplier = einops.rearrange(multiplier, "a b e -> a (b e)")
        z_dec = einops.rearrange(z, "a c d -> c (a d)").sort(dim=0).values
        z = einops.rearrange(z, "a c d -> c (a d)")
        for i in range(threshold.shape[0]):
            selected_z_dec = z_dec.index_select(dim=0, index=threshold[i, :])[0, :]
            expanded_z_dec = selected_z_dec.unsqueeze(0).expand(z.shape[0], -1)
            expanded_multiplier = multiplier[i, :].unsqueeze(0).expand(z.shape[0], -1)
            z *= torch.where(z >= expanded_z_dec, expanded_multiplier, 1.0)
        z = einops.rearrange(z, "c (a d) -> a c d", d=token_count)

    def cross_attension(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, transformer_options={}):
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
        apply_top_k_emphasis_q(sim, heads)

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
    
    def forward(x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        apply_top_k_emphasis(k, "k")
        if value is not None:
            v = self.to_v(value)
            apply_top_k_emphasis(v, "v")
            del value
        else:
            v = self.to_v(context)
            apply_top_k_emphasis(v, "v")
        if TopKEmphasis.extra_mode:
            out = cross_attension(q, k, v, self.heads, mask, transformer_options=transformer_options)
        else:
            out = attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)
    return forward

def hook_forwards(top_k_emphasis: TopKEmphasis, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(top_k_emphasis, module)
            if remove:
                del module.forward