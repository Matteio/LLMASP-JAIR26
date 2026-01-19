import requests
import time
import json
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field

@dataclass
class UsageMetadata:
    prompt_tokens: int = 0
    prompt_eval_duration: int = 0
    completion_tokens: int = 0
    eval_duration: int = 0
    total_duration: int = 0
    load_duration: int = 0
    done_reason: str = ""
    total_tokens: int = field(init=False) 

    def __post_init__(self):
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    @staticmethod
    def from_ollama_dict(data: Dict[str, Any]) -> "UsageMetadata":
        return UsageMetadata(
            prompt_tokens = data.get("prompt_eval_count", 0),
            prompt_eval_duration = data.get("prompt_eval_duration", 0),
            completion_tokens = data.get("eval_count", 0),
            eval_duration = data.get("eval_duration", 0),
            total_duration = data.get("total_duration", 0),
            load_duration = data.get("load_duration", 0),
            done_reason = data.get("done_reason", "")
        )

class LLMHandler:
    """Handler for interacting with Ollama API using requests."""
    
    def __init__(self, model_name: str="ollama", server_url: str="http://127.0.0.1:11434", 
                 api_key: str="ollama", output_format=None, timeout: int=1200, max_retries: int=4):
        """
        Initialize the LLMHandler.
        
        Args:
            model_name: The name of the model to use
            server_url: The base URL of the Ollama API
            api_key: The API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.server_url = server_url
        self.api_key = api_key
        self.model = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.output_format = output_format
        
        
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format the messages to ensure they have the correct structure."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
    
    def call(self, messages: List[Dict[str, str]], temperature: float=0, 
             stream: bool=False, max_tokens: Optional[int]=None) -> Any:
        """
        Call the Ollama API to generate a completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for sampling
            stream: Whether to stream responses
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            If stream=False: Tuple of (completion text, metadata)
            If stream=True: Generator yielding chunks of text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key != "ollama" else None
        }
        
        headers = {k: v for k, v in headers.items() if v is not None}
        

        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "stream": stream,
            "format": self.output_format 
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        endpoint = f"{self.server_url}/api/chat"
        
        if stream:
            return self._stream_response(endpoint, payload, headers)
        else:
            return self._complete_response(endpoint, payload, headers)
    
    def _stream_response(self, endpoint: str, payload: Dict[str, Any], 
                         headers: Dict[str, str]) -> str:
        """Handle streaming responses from the API."""
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = requests.post(
                    endpoint, 
                    json=payload, 
                    headers=headers, 
                    timeout=self.timeout,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if chunk:
                                return chunk  
                        except json.JSONDecodeError:
                            pass
                
                return "" 
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} retries. Error: {str(e)}")
                time.sleep(2 ** retry_count)  # Exponential backoff
    
    def _complete_response(self, endpoint: str, payload: Dict[str, Any], 
                          headers: Dict[str, str]) -> tuple:
        """Handle complete (non-streaming) responses from the API."""
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = requests.post(
                    endpoint, 
                    json=payload, 
                    headers=headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                response_data = response.json()

                print("response_data: ", response_data)
                
                completion = response_data.get('message', {}).get('content', '')
                
                meta = UsageMetadata.from_ollama_dict(response_data)

                return completion, meta
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} retries. Error: {str(e)}")
                time.sleep(2 ** retry_count)  