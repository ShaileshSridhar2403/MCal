"""
API-based models for getting next-token logits from language model APIs.
"""
import os
import numpy as np
from typing import List, Dict, Optional, Union
from openai import OpenAI


class OpenAIModel:
    """
    Wrapper for OpenAI API to extract next-token logits.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        top_logprobs: int = 5,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenAI model wrapper.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            top_logprobs: Number of top log probabilities to return (max 20)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.top_logprobs = min(top_logprobs, 20)  # OpenAI max is 20
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Uses OPENAI_API_KEY environment variable by default
            self.client = OpenAI()

    def get_next_token_logits(
        self,
        prompt: str,
        return_full_distribution: bool = False
    ) -> Dict[str, Union[str, np.ndarray, List[Dict]]]:
        """
        Get next-token logits for a given prompt.

        Args:
            prompt: Input text prompt
            return_full_distribution: If True, returns all top_logprobs for each token

        Returns:
            Dictionary containing:
                - 'generated_text': The generated response
                - 'logits': Numpy array of logits for each generated token (top token only)
                - 'top_logprobs': List of dicts with top token probabilities per position
                - 'tokens': List of generated tokens
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=self.top_logprobs,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Extract the response
        message = completion.choices[0].message
        logprobs_data = completion.choices[0].logprobs

        # Parse logprobs
        tokens = []
        logits = []
        top_logprobs_list = []

        if logprobs_data and logprobs_data.content:
            for token_data in logprobs_data.content:
                tokens.append(token_data.token)
                logits.append(token_data.logprob)

                # Extract top logprobs for this position
                if return_full_distribution and token_data.top_logprobs:
                    top_logprobs_list.append([
                        {
                            'token': tp.token,
                            'logprob': tp.logprob,
                            'bytes': tp.bytes
                        }
                        for tp in token_data.top_logprobs
                    ])

        result = {
            'generated_text': message.content,
            'logits': np.array(logits),
            'tokens': tokens,
            'top_logprobs': top_logprobs_list if return_full_distribution else None
        }

        return result

    def get_token_probabilities(
        self,
        prompt: str,
        token_position: int = 0
    ) -> Dict[str, float]:
        """
        Get probability distribution over top tokens at a specific position.

        Args:
            prompt: Input text prompt
            token_position: Position of the token to get probabilities for (0 = first generated token)

        Returns:
            Dictionary mapping tokens to their probabilities
        """
        result = self.get_next_token_logits(prompt, return_full_distribution=True)

        if result['top_logprobs'] and len(result['top_logprobs']) > token_position:
            top_logprobs = result['top_logprobs'][token_position]
            # Convert log probabilities to probabilities
            return {
                item['token']: np.exp(item['logprob'])
                for item in top_logprobs
            }

        return {}

    def predict_proba(self, messages: List[str]) -> np.ndarray:
        """
        Get averaged logits for multiple prompts (for sklearn-style interface).

        Args:
            messages: List of input prompts

        Returns:
            Numpy array of shape (n_samples, 1) with average logits
        """
        logits_list = []

        for msg in messages:
            result = self.get_next_token_logits(msg)
            # Use mean of all token logits as a simple aggregation
            avg_logit = np.mean(result['logits']) if len(result['logits']) > 0 else 0.0
            logits_list.append([avg_logit])

        return np.array(logits_list)


if __name__ == "__main__":
    # Example usage
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        logprobs=True,
        top_logprobs=2
    )

    print("Message:", completion.choices[0].message)
    print("\nLogprobs:", completion.choices[0].logprobs)

    # Using the wrapper class
    print("\n" + "="*50)
    print("Using OpenAIModel wrapper:")
    print("="*50)

    model = OpenAIModel(model_name="gpt-4o-mini", top_logprobs=5)
    result = model.get_next_token_logits("What is machine learning?", return_full_distribution=True)

    print(f"\nGenerated text: {result['generated_text']}")
    print(f"\nTokens: {result['tokens']}")
    print(f"\nLogits shape: {result['logits'].shape}")
    print(f"\nFirst 5 token logits: {result['logits'][:5]}")

    if result['top_logprobs']:
        print(f"\nTop alternatives for first token:")
        for item in result['top_logprobs'][0]:
            print(f"  {item['token']}: logprob={item['logprob']:.4f}, prob={np.exp(item['logprob']):.4f}")
