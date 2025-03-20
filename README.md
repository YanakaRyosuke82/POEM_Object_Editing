# 📜 POEM: Precise Object-level Editing via MLLM control

This project aims to allow image editing via precise instructions (e.g. move cat to the left by 12.5 px). Our method synthesizes a new image according to the editing instruction. We do this using off-the-shelf diffusion models and MLLMs with no training or fine-tuning.

<p>
  <img src="./docs/fig_method.png" alt="method" width="100%"/>
</p>
<p><i><b>Pipeline Description:</b> Given an image and an edit prompt, we first
use an MLLM to analyze the scene and identify objects. Then, we refine the detections
and enhance object masks using Grounded SAM. Next, we use a text-based LLM
to predict the transformation matrix of the initial segmentation mask. Finally, we
perform an image-to-image translation guided by the previous steps to generate the
edited image. This structured pipeline enables precise object-level editing with high
visual fidelity while preserving spatial and visual coherence.</i></p>

## 📦 Installation
### Virtual Environment Setup
```bash
module load python3/3.10.12
module load cuda
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_packages.sh
```
**Note:** The LLM **DeepseekR1-32GB** requires ~74GB VRAM.

**Dependencies:**
- **SAM2**: `python>=3.10`, `torch>=2.5.1`, `torchvision>=0.20.1`
- **QWEN-Math**: `transformers>=4.37.0`

## ⚙️ Usage
### Run Jupyter Notebook
```bash
python -m notebook --ip 0.0.0.0 --no-browser --port=8080 --allow-root
```

### Run Pipeline
```bash
python src/main.py --in_dir input_debug --out_dir output_debug --edit "grayscale"
```

## 📐 Notation
- **Image Coordinates**: Square images have the top-left at `[0, 0]` and bottom-right at `[1, 1]`.
- **Box Format**: `[Top-left x, Top-left y, Width, Height]`



![masks](./docs/fig_results.png)
We compare POEM to state-of-the-art image editing
models. We test our edit instructions using translation, scaling, appearance changing,
and a combination of them to showcase the precision of our pipeline.


## 📚 References

[1] Epstein, Dave, et al. "Diffusion self-guidance for controllable image generation." Advances in Neural Information Processing Systems 36 (2023): 16222-16239.

[2] Wu, Tsung-Han, et al. "Self-correcting llm-controlled diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.




