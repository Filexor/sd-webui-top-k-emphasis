import gradio

import modules
import modules.scripts
from modules.processing import StableDiffusionProcessing
from backend.args import dynamic_args

from scripts import prompt_parser
from scripts.classic_engine import ClassicTextProcessingEngineTopKEmphasis

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

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Top K Emphasis"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        active = gradio.Checkbox(value=False, label="Enable Top K Emphasis")
        return [active]
    
    def setup(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        if not active: return
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

    def postprocess(self, p: StableDiffusionProcessing, processed, active, *args):
        if not active: return
        match TopKEmphasis.model_type:
            case "SD1.5":
                if TopKEmphasis.text_processing_engine_original is not None:
                    p.sd_model.text_processing_engine = TopKEmphasis.text_processing_engine_original
            case "SDXL":
                if TopKEmphasis.text_processing_engine_l_original is not None:
                    p.sd_model.text_processing_engine_l = TopKEmphasis.text_processing_engine_l_original
                if TopKEmphasis.text_processing_engine_g_original is not None:
                    p.sd_model.text_processing_engine_g = TopKEmphasis.text_processing_engine_g_original