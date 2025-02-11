import logging
from typing import Dict, Any, Tuple, Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)
from ultralytics import SAM


class Models:
    def __init__(self, device_reasoning: str = "cuda", DEEP_SEEK_GPU: str = "cuda"):
        """Initialize model container"""
        self.device_reasoning = device_reasoning
        self.DEEP_SEEK_GPU = DEEP_SEEK_GPU
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

    # ======================================================================= load models ================================================================================== #
    ############ VLM 1 ############
    def load_intern_vl_2_5_38B_MPO(self) -> None:
        """
        Load InternVL2_5-38B-MPO model
        https://huggingface.co/OpenGVLab/InternVL2_5-38B-MPO
        """
        try:
            path = "OpenGVLab/InternVL2_5-38B-MPO"
            model = (
                AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
            )
            self.models["intern_vl_2_5_38B_MPO"] = model
            logging.info("Loaded intern_vl_2_5_38B_MPO model successfully")
        except Exception as e:
            logging.error(f"Failed to load intern_vl_2_5_38B_MPO model: {e}")
            raise

    ############ VLM 2 ############
    def load_qwen_2_5_vl_7b(self) -> None:
        """Load Qwen VL model and processor"""
        try:
            # Automatically map model to available devices
            self.models["qwen_2_5_vl_7b"] = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map={"": self.device_reasoning}
            )
            self.processors["qwen_2_5_vl_7b"] = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            logging.info("Loaded Qwen VL model successfully")
        except Exception as e:
            logging.error(f"Failed to load Qwen VL model: {e}")
            raise

    ############ VLM 3 ############
    def load_ovis1_6_gemma2_27B(self) -> None:
        """Load Ovis1.6-Gemma2-27B model"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "AIDC-AI/Ovis1.6-Gemma2-27B", torch_dtype=torch.bfloat16, multimodal_max_length=8192, trust_remote_code=True
            ).to(self.device_reasoning)
            text_tokenizer = model.get_text_tokenizer()

            self.models["ovis1_6_gemma2_27B"] = (model, text_tokenizer)
            logging.info("Loaded ovis1_6_gemma2_27B model successfully")
        except Exception as e:
            logging.error(f"Failed to load ovis1_6_gemma2_27B model: {e}")
            raise

    ############ math-LLM 1 ############
    def load_qwen2_5_math_7b_instruct(self) -> None:
        """Initialize the language model for transformation reasoning."""
        try:
            model_name = "Qwen/Qwen2.5-Math-7B-Instruct"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.DEEP_SEEK_GPU},
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.models["qwen2_5_math_7b_instruct"] = (model, tokenizer)
            logging.info("Loaded Qwen Math model successfully")
        except Exception as e:
            logging.error(f"Failed to load Qwen Math model: {e}")
            raise

    ############ math-LLM 2 ############
    def load_deepseek_r1_distill_qwen_32B(self) -> None:
        """Initialize the language model for text processing."""
        try:
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.DEEP_SEEK_GPU},
            )
            print(model.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.models["deepseek_r1_distill_qwen_32B"] = (model, tokenizer)
            logging.info("Loaded DeepSeek R1 Distill Qwen 32B model successfully")
        except Exception as e:
            logging.error(f"Failed to load DeepSeek R1 Distill Qwen 32B model: {e}")
            raise

    def load_sam(self) -> None:
        """Load SAM model"""
        try:
            self.models["sam"] = SAM("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/sam2.1_b.pt")
            logging.info("Loaded SAM model successfully")
        except Exception as e:
            logging.error(f"Failed to load SAM model: {e}")
            raise

    # ======================================================================= get models ================================================================================== #
    ############ VLM 1 ############
    def get_intern_vl_2_5_38B_MPO(self):
        """Get InternVL2_5-38B-MPO model"""
        if "intern_vl_2_5_38B_MPO" not in self.models:
            self.load_intern_vl_2_5_38B_MPO()
        return self.models["intern_vl_2_5_38B_MPO"]

    ############ VLM 2 ############
    def get_qwen_2_5_vl_7b(self):
        """Get Qwen model and processor"""
        if "qwen_2_5_vl_7b" not in self.models:
            self.load_qwen_2_5_vl_7b()
        return self.models["qwen_2_5_vl_7b"], self.processors["qwen_2_5_vl_7b"]

    ############ VLM 3 ############
    def get_ovis1_6_gemma2_27B(self):
        """Get Ovis1.6-Gemma2-27B model"""
        if "ovis1_6_gemma2_27B" not in self.models:
            self.load_ovis1_6_gemma2_27B()
        return self.models["ovis1_6_gemma2_27B"]

    ############ math-LLM 1 ############
    def get_qwen2_5_math_7b_instruct(self):
        """Get Qwen Math model"""
        if "qwen2_5_math_7b_instruct" not in self.models:
            self.load_qwen2_5_math_7b_instruct()
        return self.models["qwen2_5_math_7b_instruct"]

    ############ math-LLM 2 ############
    def get_deepseek_r1_distill_qwen_32B(self):
        """Get DeepSeek R1 Distill Qwen 32B model"""
        if "deepseek_r1_distill_qwen_32B" not in self.models:
            self.load_deepseek_r1_distill_qwen_32B()
        return self.models["deepseek_r1_distill_qwen_32B"]

    def get_sam(self):
        """Get SAM model"""
        if "sam" not in self.models:
            self.load_sam()
        return self.models["sam"]


if __name__ == "__main__":
    # Create a new logger for model loading with colored output
    model_loader_logger = logging.getLogger("model_loader")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - \033[37m%(message)s\033[0m")
    handler.setFormatter(formatter)
    model_loader_logger.addHandler(handler)
    model_loader_logger.setLevel(logging.INFO)

    model_loader_logger.info("Loading InternVL2_5-38B-MPO model...")
    models = Models()
    models.load_intern_vl_2_5_38B_MPO()
    del models

    model_loader_logger.info("Loading Qwen VL 7B model...")
    models = Models()
    models.load_qwen_2_5_vl_7b()
    del models

    model_loader_logger.info("Loading Ovis1.6-Gemma2-27B model...")
    models = Models()
    models.load_ovis1_6_gemma2_27B()
    del models

    model_loader_logger.info("Loading Qwen Math 7B model...")
    models = Models()
    models.load_qwen2_5_math_7b_instruct()
    del models

    model_loader_logger.info("Loading DeepSeek R1 Distill Qwen 32B model...")
    models = Models()
    models.load_deepseek_r1_distill_qwen_32B()
    del models

    model_loader_logger.info("Loading SAM model...")
    models = Models()
    models.load_sam()
    del models
