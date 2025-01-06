# VLM_controller_for_SD

### To Do List

Please let me work on the pipeline so I can improve my code skills and learn.
I gave you some parallel task you can work independerntly on 

#### Marco's Tasks:
- [ ] Set up the complete pipeline top-down with all modules:
  - VLM parser for reasoning
    - download QWEN -- takes ages.
  - SDL drawer for image generation
  - OpenCV shape editor for transformations (already partly done by Onur) 

#### Onur's Tasks:
- [ ] Generate synthetic dataset for testing non-affine transformations. 
- [ ] re-think / define edit prompts for each non-affine category

if the test dataset fails, than we can try fine-tuning QWEN.

#### Together (at last)
- [ ] Integrate NVIDIA's Add-it module into pipeline for improved object insertion
- [ ] download magicbrush and run/evaluate our pipeline on those testsevaluate
- [ ] ablation studies compare "SDL" with "Self-Guidance" drawers (not that important)


----
----


### Installation
venv installation
```
python3 -m venv ./.venv/
source ./.venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install notebook ipywidgets
pip install --upgrade transformers>=4.37.0
```

setup QWEN-2.5.Math:  
```
pip install --upgrade transformers>=4.37.0      
# https://github.com/QwenLM/Qwen2.5-Math
```

setup Qwen-VL
```
git clone git@github.com:QwenLM/Qwen-VL.git src/
pip install -r src/Qwen-VL/requirements.txt
```

setup jupyter notebook from a secondary tmux session
```
python -m notebook --ip 0.0.0.0 --no-browser --port=8080 --allow-root
```



### Introduction
We aim to leverage the prompting of large foundation models, such as stable diffusion, to enhance 2D image editing through a human-machine collaboration methodology. Our research focuses on the test-time adaptation of large generative models and human-in-the-loop methods in computer vision.
The goal of the project is to develop a "reasoning module" that interprets user prompts to extract mathematically precise parameters for both affine and non-affine transformations used in shape editing.
     
The novelty comes in three parts:
1. the visual reasoning of user edit prompts for parsing of almost-mathematically precise shape edits.
2. the visual reasoning is based exclusively on the source image, without relying on text descriptions.
3. a comprehensive pipeline that includes leverages our improved reasoning module together with a drawing module for any type of edit.


### Modelling
This project builds upon the popular VLM-based pipeline, wherein a Vision-Language Model (VLM) aids in reasoning based on the user's input prompt, followed by a Stable Diffusion model acting as the drawing agent. 

Our goal is to enhance this pipeline by refining the "reasoning phase" prior to the drawing phase. The ehancement wouild enable to have almost matematicallty precise shape edits. The output from the reasoning module will generate detailed information about (a) shape transformations and (b) appearance transformations of the main objects, leading to a holistic system. This output will then be processed using the SDL method [^2] inspired by self-guidance formulation[^1], with an extension of the NVIDIA-addit [^3] module specifically designed for improving the location of the "add object" operation.

![alt text](./docs/approach.png)

***General Idea:** We aim to separate the reasoning process (Phase I) from the drawing phase (Phase II). Our contribution focuses on enhancing the Vision-Language Model (VLM) parsing of user prompts by extracting mathematical parameters that will be used to create masks for transformations, including both affine and non-affine types.*



![alt text](./docs/detailed_pipeline.png)
***Detailed Pipeline:** The source image is processed by a mathematically robust VLM (e.g., Qwen), which outputs a list of properties (mathematical parameters) related to appearance and shape edits. We then create a binary edit mask using the extracted parameters, which guides the drawing model (e.g., the latest version of Stable Diffusion) during the inference phase based on self-guidance equations[^3].*



### References:
[^1]: Epstein, Dave, et al. "Diffusion self-guidance for controllable image generation." Advances in Neural Information Processing Systems 36 (2023): 16222-16239.

[^2]: Wu, Tsung-Han, et al. "Self-correcting llm-controlled diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[^3]: Tewel, Yoad, et al. "Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models." arXiv preprint arXiv:2411.07232 (2024).
