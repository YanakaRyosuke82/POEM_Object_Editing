import requests
import torch
from io import BytesIO
from PIL import Image  # ✅ Fix import
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS

# help(StableDiffusionPipeline_LEDITS.invert)

model = 'runwayml/stable-diffusion-v1-5'
device = 'cuda'

pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model, safety_checker=None)
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
    model, subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2
)
pipe.to(device)

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")  # ✅ Use PIL's Image

gen = torch.Generator(device=device)
gen.manual_seed(21)

img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/cherry_blossom.png"
image = download_image(img_url)
image.save("temp.jpg")  # ✅ Save the image first

# Fix function call (use correct args)
_ = pipe.invert(image_path="temp.jpg", num_inversion_steps=50, skip=0.1)

edited_image = pipe(
    editing_prompt=["move car left"],
    edit_guidance_scale=5.0,
    edit_threshold=0.75,
).images[0]

edited_image.save("edited_image.jpg")  # ✅ Save edited image to disk
