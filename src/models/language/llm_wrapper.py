"""Language model wrappers for LLaMA, Falcon, Mistral, etc."""

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class LLMWrapper(BaseModel):
    """Generic wrapper for large language models."""
    
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        max_length: int = 512,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize LLM wrapper.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            num_classes: Number of output classes
            max_length: Maximum sequence length
            model_name: Custom model name
            device: Device to run model on
            load_in_8bit: Whether to load model in 8-bit
            load_in_4bit: Whether to load model in 4-bit
        """
        if model_name is None:
            model_name = f"llm_{model_name_or_path.split('/')[-1]}"
        
        super().__init__(num_classes, model_name, device)
        
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.classification_head = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the language model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            model_kwargs = {}
            if self.load_in_8bit:
                model_kwargs['load_in_8bit'] = True
            elif self.load_in_4bit:
                model_kwargs['load_in_4bit'] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                **model_kwargs
            )
            
            # Add classification head
            hidden_size = self.model.config.hidden_size
            self.classification_head = nn.Linear(hidden_size, self.num_classes)
            
            if not (self.load_in_8bit or self.load_in_4bit):
                self.model = self.model.to(self.device)
                self.classification_head = self.classification_head.to(self.device)
            
            logger.info(f"Loaded LLM: {self.model_name_or_path}")
            
        except ImportError:
            logger.error("transformers library not found. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load LLM {self.model_name_or_path}: {e}")
            raise
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input texts.
        
        Args:
            texts: Input text(s)
            add_special_tokens: Whether to add special tokens
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through LLM with classification head.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Classification logits
        """
        # Get last hidden states from LLM
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use last token for classification (or mean pooling)
        if attention_mask is not None:
            # Get the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            last_hidden = hidden_states[range(batch_size), sequence_lengths]
        else:
            # Use last token
            last_hidden = hidden_states[:, -1, :]
        
        # Classification
        logits = self.classification_head(last_hidden)
        
        return logits
    
    def predict_proba(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Get probability predictions for texts.
        
        Args:
            texts: Input text(s)
            
        Returns:
            Probability predictions
        """
        self.eval()
        
        # Tokenize inputs
        inputs = self.tokenize(texts)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
        
        return probs
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        self.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text (remove input prompt)
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return generated_text


class LlamaWrapper(LLMWrapper):
    """Wrapper specifically for LLaMA models."""
    
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        num_classes: int = 4,
        max_length: int = 512,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize LLaMA wrapper."""
        if model_name is None:
            model_name = f"llama_{model_name_or_path.split('/')[-1]}"
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            num_classes=num_classes,
            max_length=max_length,
            model_name=model_name,
            device=device,
            **kwargs
        )


class FalconWrapper(LLMWrapper):
    """Wrapper specifically for Falcon models."""
    
    def __init__(
        self,
        model_name_or_path: str = "tiiuae/falcon-7b",
        num_classes: int = 4,
        max_length: int = 512,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize Falcon wrapper."""
        if model_name is None:
            model_name = f"falcon_{model_name_or_path.split('/')[-1]}"
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            num_classes=num_classes,
            max_length=max_length,
            model_name=model_name,
            device=device,
            **kwargs
        )


class MistralWrapper(LLMWrapper):
    """Wrapper specifically for Mistral models."""
    
    def __init__(
        self,
        model_name_or_path: str = "mistralai/Mistral-7B-v0.1",
        num_classes: int = 4,
        max_length: int = 512,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize Mistral wrapper."""
        if model_name is None:
            model_name = f"mistral_{model_name_or_path.split('/')[-1]}"
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            num_classes=num_classes,
            max_length=max_length,
            model_name=model_name,
            device=device,
            **kwargs
        )


# Registry of language models
LANGUAGE_MODELS = {
    "llama": LlamaWrapper,
    "falcon": FalconWrapper,
    "mistral": MistralWrapper,
    "generic_llm": LLMWrapper,
}


def get_language_model(model_name: str, **kwargs) -> LLMWrapper:
    """Get a language model by name.
    
    Args:
        model_name: Name of the language model
        **kwargs: Additional arguments
        
    Returns:
        Language model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in LANGUAGE_MODELS:
        raise ValueError(
            f"Unknown language model: {model_name}. "
            f"Available models: {list(LANGUAGE_MODELS.keys())}"
        )
    
    return LANGUAGE_MODELS[model_name](**kwargs)