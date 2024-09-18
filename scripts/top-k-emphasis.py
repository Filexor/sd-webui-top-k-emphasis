import gradio
import modules
import modules.scripts
from modules.processing import StableDiffusionProcessing

from scripts import prompt_parser

filtering_locations = ["Text Processing", "After K", "After QK", "After QKV"]
zero_point = ["Zero", "Mean of Prompt", "Median of Prompt", "Mean of Padding", "Median of padding"]

class Script(modules.scripts.Script):
    """
    Applies emphasis at following locations:
        "c": After conditioning has been constructed.
        "b": Before every k in cross attention.
        "k": After every k in cross attention.
        "q": After every q * v in cross attention.
        "s": After every softmax in cross attention.
        "v": After every v in cross attention.
    """
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Top K Emphasis"
    
    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        active = gradio.Checkbox(value=False, label="Enable Top K Emphasis")
        return [active]
    
    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        if not active: return
        p.c = prompt_parser.get_multicond_learned_conditioning(p.sd_model, p.prompts, p.steps)