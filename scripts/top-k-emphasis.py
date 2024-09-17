import gradio
import modules
import modules.scripts

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
        active = gradio.Checkbox(Value=False, label="Enable Top K Emphasis")
        return [active]
    
    def process_batch(self, p, active, *args, **kwargs):
        if not active: return
