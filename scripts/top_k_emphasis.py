from math import e
import gradio
from regex import W
import torch
import einops

from backend.operations import weights_manual_cast
import modules
from modules.devices import device
from modules.processing import StableDiffusionProcessing
import modules.scripts
import modules.script_callbacks
from modules import sd_samplers
from backend import memory_management
from backend.args import args, dynamic_args
from backend.attention import attention_function

from scripts import emphasis
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
        "q": After every matmul(q, k) in cross attention.
        "s": Similar to "q" but after softmax.
    """

    active = False
    extra_mode = False
    model_type = None
    text_processing_engine_original = None
    text_processing_engine_l_original = None
    text_processing_engine_g_original = None
    get_learned_conditioning_sdxl_original = None
    positive_multiplier = None
    negative_multiplier = None
    reconstructed_positive_multiplier = None
    reconstructed_negative_multiplier = None
    crossattentioncounter = 0
    emphasis_view_update = False
    debug = False

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
            with gradio.Row():
                manual_mode = gradio.Checkbox(value=False, label="Disallow to automatically add start token")
                emphasis_view_update = gradio.Checkbox(value=False, label="Update view on every emphasis")
            with gradio.Row():
                debug = gradio.Checkbox(value=False, label="Debug mode")
        return [active, extra_mode, manual_mode, emphasis_view_update, debug]
    
    def before_process(self, p: StableDiffusionProcessing, active, extra_mode, manual_mode, emphasis_view_update, debug, *args, **kwargs):
        TopKEmphasis.active = active
        TopKEmphasis.extra_mode = extra_mode
        if not active: return
        print("Loading Top K Emphasis.")
        if hasattr(p.sd_model, "text_processing_engine"):
            TopKEmphasis.model_type = "SD1.5"
            TopKEmphasis.text_processing_engine_original = p.sd_model.text_processing_engine
            if not isinstance(p.sd_model.text_processing_engine, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine.tokenizer,
                    embeddings=p.sd_model.text_processing_engine.embeddings,
                    embedding_key='clip_l',
                    token_embedding=p.sd_model.text_processing_engine.text_encoder.transformer.text_model.embeddings.token_embedding,
                    text_projection=False,
                    minimal_clip_skip=1,
                    clip_skip=1,
                    return_pooled=False,
                    final_layer_norm=True,
                    manual_mode=manual_mode,
                    emphasis_view_update=emphasis_view_update,
                    debug=debug,
                )
            hook_forwards(p.sd_model.forge_objects.unet.model, False)
            TopKEmphasis.emphasis_view_update = emphasis_view_update
            TopKEmphasis.debug = debug
        elif hasattr(p.sd_model, "text_processing_engine_l") and hasattr(p.sd_model, "text_processing_engine_g"):
            TopKEmphasis.model_type = "SDXL"
            TopKEmphasis.text_processing_engine_l_original = p.sd_model.text_processing_engine_l
            if not isinstance(p.sd_model.text_processing_engine_l, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine_l = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine_l.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine_l.tokenizer,
                    embeddings=p.sd_model.text_processing_engine_l.embeddings,
                    embedding_key='clip_l',
                    token_embedding=p.sd_model.text_processing_engine_l.text_encoder.transformer.text_model.embeddings.token_embedding,
                    text_projection=False,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=False,
                    final_layer_norm=False,
                    manual_mode=manual_mode,
                    emphasis_view_update=emphasis_view_update,
                    debug=debug,
                )
            TopKEmphasis.text_processing_engine_g_original = p.sd_model.text_processing_engine_g
            if not isinstance(p.sd_model.text_processing_engine_g, ClassicTextProcessingEngineTopKEmphasis):
                p.sd_model.text_processing_engine_g = ClassicTextProcessingEngineTopKEmphasis(
                    text_encoder=p.sd_model.text_processing_engine_g.text_encoder,
                    tokenizer=p.sd_model.text_processing_engine_g.tokenizer,
                    embeddings=p.sd_model.text_processing_engine_g.embeddings,
                    embedding_key='clip_g',
                    token_embedding=p.sd_model.text_processing_engine_g.text_encoder.transformer.text_model.embeddings.token_embedding,
                    text_projection=True,
                    minimal_clip_skip=2,
                    clip_skip=2,
                    return_pooled=True,
                    final_layer_norm=False,
                    manual_mode=manual_mode,
                    emphasis_view_update=emphasis_view_update,
                    debug=debug,
                )
            TopKEmphasis.get_learned_conditioning_sdxl_original = p.sd_model.get_learned_conditioning
            p.sd_model.get_learned_conditioning = get_learned_conditioning_sdxl
            hook_forwards(p.sd_model.forge_objects.unet.model, False)
            TopKEmphasis.emphasis_view_update = emphasis_view_update
            TopKEmphasis.debug = debug
        else:
            raise Exception("Unsupported model type.")

    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        if not active: return
        p.setup_conds = lambda: None
        p.cached_c = [None, None, None]
        p.cached_uc = [None, None, None]
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
                if TopKEmphasis.get_learned_conditioning_sdxl_original is not None:
                    p.sd_model.get_learned_conditioning = TopKEmphasis.get_learned_conditioning_sdxl_original
                hook_forwards(p.sd_model.forge_objects.unet.model, True)
        print("Unloading Top K Emphasis.")

def top_k_emphasis_on_cfg_denoiser(params: modules.script_callbacks.CFGDenoiserParams, *args, **kwargs):
    if not TopKEmphasis.active: return
    TopKEmphasis.reconstructed_positive_multiplier = prompt_parser.reconstruct_multi_multiplier_batch(TopKEmphasis.positive_multiplier, params.sampling_step)
    TopKEmphasis.reconstructed_negative_multiplier = prompt_parser.reconstruct_multiplier_batch(TopKEmphasis.negative_multiplier, params.sampling_step) if TopKEmphasis.negative_multiplier is not None else None
    TopKEmphasis.crossattentioncounter = 0

modules.script_callbacks.on_cfg_denoiser(top_k_emphasis_on_cfg_denoiser)

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
        uc, um = None, None
        print('Skipping unconditional conditioning when CFG = 1. Negative Prompts are ignored.')
    else:
        uc, um = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)

    c, m = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)
    return c, m, uc, um

def to_structure_of_tensor(positive: list[EmphasisPair], negative: list[EmphasisPair], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    weights_p = []
    thresholds_p = []
    weights_n = []
    thresholds_n = []
    for i in positive:
        weight = []
        threshold = []
        count_tokenwise = 0
        for j in i.multipliers:
            if j.key == key:
                count_tokenwise += 1
                weight.append(j.weight)
                threshold.append(j.threshold)
        count = max(count, count_tokenwise)
        weights_p.append(weight)
        thresholds_p.append(threshold)
    if negative is not None:
        for i in negative:
            weight = []
            threshold = []
            count_tokenwise = 0
            for j in i.multipliers:
                if j.key == key:
                    count_tokenwise += 1
                    weight.append(j.weight)
                    threshold.append(j.threshold)
            count = max(count, count_tokenwise)
            weights_n.append(weight)
            thresholds_n.append(threshold)
    for i in weights_p:
        i += [1.0] * (count - len(i))
    for i in thresholds_p:
        i += [0.0] * (count - len(i))
    if negative is not None:
        for i in weights_n:
            i += [1.0] * (count - len(i))
        for i in thresholds_n:
            i += [0.0] * (count - len(i))
    weight_p = torch.asarray(weights_p)
    weight_p = einops.rearrange(weight_p, "a b -> b a")
    threshold_p = torch.asarray(thresholds_p)
    threshold_p = einops.rearrange(threshold_p, "a b -> b a")
    if negative is not None:
        weight_n = torch.asarray(weights_n)
        weight_n = einops.rearrange(weight_n, "a b -> b a")
        threshold_n = torch.asarray(thresholds_n)
        threshold_n = einops.rearrange(threshold_n, "a b -> b a")
    if negative is not None:
        return torch.stack((weight_n, weight_p), dim=1), torch.stack((threshold_n, threshold_p), dim=1)
    else:
        return torch.stack((weight_p, ), dim=1), torch.stack((threshold_p, ), dim=1)

def hook_forward(self):
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
    
    def apply_top_k_emphasis1(z: torch.Tensor, key: str):
        if key == "k":
            multiplier = TopKEmphasis.current_step_k_mul
            threshold = TopKEmphasis.current_step_k_thres   # [depth, c/uc, token]
        elif key == "v":
            multiplier = TopKEmphasis.current_step_v_mul
            threshold = TopKEmphasis.current_step_v_thres   # [depth, c/uc, token]
        token_count = multiplier.shape[-1]
        threshold = torch.where(threshold == 0.0, z.shape[2], threshold)
        threshold = torch.where(threshold < 1.0, threshold * z.shape[2], threshold)
        threshold = threshold - 1
        threshold = threshold.to(dtype=torch.int32)
        threshold = einops.rearrange(threshold, "a b e -> a (b e)").to(device)
        multiplier = einops.rearrange(multiplier, "a b e -> a (b e)").to(device)
        z = einops.rearrange(z, "a c d -> d (a c)")
        for i in range(threshold.shape[0]):
            z_dec = z.sort(dim=0, descending=True).values
            selected_z_dec = z_dec.index_select(dim=0, index=threshold[i, :]).diag()
            expanded_z_dec = selected_z_dec.unsqueeze(0).expand(z.shape[0], -1)
            expanded_multiplier = multiplier[i, :].unsqueeze(0).expand(z.shape[0], -1)
            z *= torch.where(z >= expanded_z_dec, expanded_multiplier, 1.0)
        z = einops.rearrange(z, "d (a c) -> a c d", c=token_count)
        return z

    def apply_top_k_emphasis2(z: torch.Tensor, key: str, heads):
        if key == "q":
            multiplier = TopKEmphasis.current_step_q_mul
            threshold = TopKEmphasis.current_step_q_thres   # [depth, c/uc, token]
        if key == "s":
            multiplier = TopKEmphasis.current_step_s_mul
            threshold = TopKEmphasis.current_step_s_thres   # [depth, c/uc, token]
        token_count = multiplier.shape[-1]
        z = einops.rearrange(z, "(a b) c d -> (b c) (a d)", b=heads)
        threshold = torch.where(threshold == 0.0, z.shape[0], threshold)
        threshold = torch.where(threshold < 1.0, threshold * z.shape[0], threshold)
        threshold = threshold - 1
        threshold = threshold.to(dtype=torch.int32)
        threshold = threshold[:, :, :].to(device)
        multiplier = multiplier[:, :, :].to(device)
        threshold = einops.rearrange(threshold, "a b e -> a (b e)")
        multiplier = einops.rearrange(multiplier, "a b e -> a (b e)")
        for i in range(threshold.shape[0]):
            z_dec = z.sort(dim=0, descending=True).values
            selected_z_dec = z_dec.index_select(dim=0, index=threshold[i, :]).diag()
            expanded_z_dec = selected_z_dec.unsqueeze(0).expand(z.shape[0], -1)
            expanded_multiplier = multiplier[i, :].unsqueeze(0).expand(z.shape[0], -1)
            z *= torch.where(z >= expanded_z_dec, expanded_multiplier, 1.0)
        z = einops.rearrange(z, "(b c) (a d) -> (a b) c d", b=heads, d=token_count)
        return z

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
        sim = einops.rearrange(sim, "(i h) l t -> i t (h l)", h=heads)
        sim = emphasis.emphasis_crossattention(sim, TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "q", 
                                    TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
        sim = einops.rearrange(sim, "i t (h l) -> (i h) l t", h=heads)

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
        sim = einops.rearrange(sim, "(i h) l t -> i t (h l)", h=heads)
        sim = emphasis.emphasis_crossattention(sim, TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "s", 
                                    TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
        sim = einops.rearrange(sim, "i t (h l) -> (i h) l t", h=heads)
        out = torch.einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out
    
    def forward(x, context: torch.Tensor=None, value=None, mask=None, transformer_options={}):
        # context is ordered in (uncond_batch0, uncond_batch1, ..., cond_batch0_and-1, cond_batch1_and-1, ..., cond_batch0_and-2, ...) .
        # if no uncond, context is ordered in (cond_batch0_and-1, cond_batch1_and-1, ..., cond_batch0_and-2, ...) .
        # batch_size is z.shape[0] / (len(uncond_indices) + len(cond_indicies))
        # check uncond_indices and cond_indicies in transformer_options .
        # context = "(uncond_cond batch) token channel"
        # k = "(uncond_cond batch) token channel"
        # note that uncond_cond can be non-rectangular.
        # if size of tokens is not uniform, last token of short tensor will be broadcasted. i do not care such case.
        #[[..., 74, 75, 76, 77, 78, 79, ...],
        # [..., 74, 75, 76, 76, 76, 76, ...]]
        q = self.to_q(x)
        context = default(context, x)
        context_k = emphasis.emphasis_crossattention(context.clone(), TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "pk", 
                                  TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
        k = self.to_k(context_k)
        del context_k
        k = emphasis.emphasis_crossattention(k, TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "k", 
                                  TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
        if value is not None:
            value_v = emphasis.emphasis_crossattention(value.clone(), TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "pv", 
                                  TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
            v = self.to_v(value_v)
            del value_v
            v = emphasis.emphasis_crossattention(v, TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "v", 
                                    TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
            del value
        else:
            context_v = emphasis.emphasis_crossattention(context.clone(), TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "pv", 
                                  TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
            v = self.to_v(context_v)
            del context_v
            v = emphasis.emphasis_crossattention(v, TopKEmphasis.reconstructed_positive_multiplier, TopKEmphasis.reconstructed_negative_multiplier, "v", 
                                    TopKEmphasis.crossattentioncounter, TopKEmphasis.emphasis_view_update, TopKEmphasis.debug)
        if TopKEmphasis.extra_mode:
            out = cross_attension(q, k, v, self.heads, mask, transformer_options=transformer_options)
        else:
            out = attention_function(q, k, v, self.heads, mask)
        TopKEmphasis.crossattentioncounter += 1
        return self.to_out(out)
    return forward

def hook_forwards(root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(module)
            if remove:
                del module.forward