from lmdeploy import pipeline
from PIL import Image

# Load the image using PIL
image_path = "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/benchmark/input/2007_000039_0_transform_0_prompt_0/input_image.png"
image = Image.open(image_path)

# pipe = pipeline("OpenGVLab/InternVL2_5-26B")  #  ok fits
pipe = pipeline("OpenGVLab/InternVL2_5-8B")  #  ok fits

# OpenAI-style prompt format
prompts = [{"role": "user", "content": [{"type": "text", "text": "describe this image"}, {"type": "image_url", "image_url": {"url": image}}]}]
response = pipe(prompts)
print(response.text)

# Another prompt example
prompts = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What objects do you see in this image? List them with their locations."},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }
]
response = pipe(prompts)
print(response.text)
