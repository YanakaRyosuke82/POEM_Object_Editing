import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import logging
from ultralytics import SAM
class Models:
    def __init__(self, device_reasoning: str = "cuda", DEEP_SEEK_GPU: str = "cuda"):
        """Initialize model container"""
        self.device_reasoning = device_reasoning
        self.DEEP_SEEK_GPU = DEEP_SEEK_GPU
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
    # ======================================================================= load models ================================================================================== #
    def load_qwen_vlm(self) -> None:
        """Load Qwen VL model and processor"""
        try:
            # Automatically map model to available devices
            self.models['qwen'] = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map = {"": self.device_reasoning}
            )
            self.processors['qwen'] = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            logging.info("Loaded Qwen VL model successfully")
        except Exception as e:
            logging.error(f"Failed to load Qwen VL model: {e}")
            raise
            
    def load_sam(self) -> None:
        """Load SAM model"""
        try:
            self.models['sam'] = SAM("/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/sam2.1_b.pt")
            logging.info("Loaded SAM model successfully")
        except Exception as e:
            logging.error(f"Failed to load SAM model: {e}")
            raise
        
    def load_qwen_math(self) -> None:
        """Initialize the language model for transformation reasoning."""
        try:
            model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device_reasoning,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.models['qwen_math'] = (model, tokenizer)
            logging.info("Loaded Qwen Math model successfully")
        except Exception as e:
            logging.error(f"Failed to load Qwen Math model: {e}")
            raise
            
    def load_deepseek_r1_text(self) -> None:
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
            
            self.models['deepseek_r1_text'] = (model, tokenizer)
            logging.info("Loaded DeepSeek R1 Text model successfully")
        except Exception as e:
            logging.error(f"Failed to load DeepSeek R1 Text model: {e}")
            raise
    # ======================================================================= get models ================================================================================== #
    def get_qwen_vlm(self):
        """Get Qwen model and processor"""
        if 'qwen' not in self.models:
            self.load_qwen_vlm()
        return self.models['qwen'], self.processors['qwen']
        
    def get_sam(self):
        """Get SAM model"""
        if 'sam' not in self.models:
            self.load_sam()
        return self.models['sam'] 
    
    def get_qwen_math(self):
        """Get Qwen Math model"""
        if 'qwen_math' not in self.models:
            self.load_qwen_math()
        return self.models['qwen_math']
    
    def get_deepseek_r1_text(self):
        """Get DeepSeek R1 Text model"""
        if 'deepseek_r1_text' not in self.models:
            self.load_deepseek_r1_text()
        return self.models['deepseek_r1_text']
    
