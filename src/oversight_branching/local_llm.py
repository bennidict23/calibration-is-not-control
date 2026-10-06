from dataclasses import dataclass
from typing import List, Optional

import os
import requests


@dataclass
class GenerationOutput:
    prompt: str
    text: str


class LocalChatModel:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 2048,
        api_base: Optional[str] = None,
        api_model: Optional[str] = None,
        api_timeout: float = 60.0,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_model = api_model or model_path
        self.api_timeout = api_timeout
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = None
        self.session = requests.Session() if self.api_base else None
        if self.session is not None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.llm = None
        self.sampling_params_cls = None
        self.remote_mode = "completions"
        if self.api_base is not None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except (ImportError, OSError):
                self.tokenizer = None
                self.remote_mode = "chat_completions"
        if self.api_base is None:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=True,
            )
            self.sampling_params_cls = SamplingParams
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _post_with_retry(self, url, json_data, max_retries=5):
        import time
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=json_data, timeout=self.api_timeout)
                if response.status_code in (502, 503, 429, 500):
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    wait = 2 ** attempt * 5
                    print(f"[retry {attempt+1}/{max_retries}] HTTP {response.status_code}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                return response
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt * 5
                print(f"[retry {attempt+1}/{max_retries}] {type(e).__name__}, waiting {wait}s...")
                time.sleep(wait)

    def _raise_remote_error(self, response, prompt):
        prompt_tokens = None
        if self.tokenizer is not None:
            try:
                prompt_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            except Exception:
                prompt_tokens = None
        body = response.text[:500].replace("\n", " ")
        token_note = f", prompt_tokens={prompt_tokens}" if prompt_tokens is not None else ""
        message = f"HTTP {response.status_code} for {response.url}{token_note}, body={body}"
        raise requests.exceptions.HTTPError(message, response=response)

    def _generate_remote(self, rendered: List[str], token_limit: int, stop: Optional[List[str]]):
        outputs = []
        for prompt in rendered:
            if self.remote_mode == "completions":
                response = self._post_with_retry(
                    f"{self.api_base}/v1/completions",
                    {
                        "model": self.api_model,
                        "prompt": prompt,
                        "temperature": 0.0,
                        "max_tokens": token_limit,
                        "stop": stop,
                    },
                )
                if response.status_code != 404:
                    if not response.ok:
                        self._raise_remote_error(response, prompt)
                    payload = response.json()
                    outputs.append(payload["choices"][0]["text"])
                    continue
                self.remote_mode = "chat_completions"

            chat_payload = {
                "model": self.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": token_limit,
                "stop": stop,
            }
            response = self._post_with_retry(
                f"{self.api_base}/v1/chat/completions",
                chat_payload,
            )
            # Newer OpenAI chat-completions models reject max_tokens in favor of
            # max_completion_tokens. Retry once with the supported field.
            if (
                response.status_code == 400
                and "max_completion_tokens" in response.text
                and "max_tokens" in chat_payload
            ):
                chat_payload.pop("max_tokens", None)
                chat_payload["max_completion_tokens"] = token_limit
                response = self._post_with_retry(
                    f"{self.api_base}/v1/chat/completions",
                    chat_payload,
                )
            if (
                response.status_code == 400
                and "\"stop\"" in response.text
                and chat_payload.get("stop") is not None
            ):
                chat_payload.pop("stop", None)
                response = self._post_with_retry(
                    f"{self.api_base}/v1/chat/completions",
                    chat_payload,
                )
            if not response.ok:
                self._raise_remote_error(response, prompt)
            payload = response.json()
            outputs.append(payload["choices"][0]["message"]["content"])
        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        max_new_tokens: int = None,
        assistant_prefix: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> List[GenerationOutput]:
        outputs: List[GenerationOutput] = []
        token_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            rendered = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) if self.tokenizer is not None else prompt
                for prompt in batch_prompts
            ]
            # In completions mode (local vLLM or remote with tokenizer),
            # assistant_prefix is appended to the rendered prompt so the model
            # continues from that prefix.  In chat_completions mode the prefix
            # is included in the user message instead, because the API controls
            # the assistant turn boundary.
            using_chat = (self.api_base is not None and self.remote_mode == "chat_completions")
            if assistant_prefix and not using_chat:
                rendered = [text + assistant_prefix for text in rendered]
            if assistant_prefix and using_chat:
                rendered = [
                    text + f"\nRespond starting with: {assistant_prefix}"
                    for text in rendered
                ]
            if self.api_base is None:
                generated_text = [
                    output.outputs[0].text
                    for output in self.llm.generate(
                        rendered,
                        self.sampling_params_cls(
                            temperature=0.0,
                            max_tokens=token_limit,
                            stop=stop,
                        ),
                    )
                ]
            else:
                generated_text = self._generate_remote(rendered, token_limit, stop)
            for prompt, text_out in zip(batch_prompts, generated_text):
                # In completions mode the model output does not include the
                # prefix, so we prepend it.  In chat_completions mode the model
                # typically echoes the prefix itself, so we do not double it.
                if using_chat:
                    text = text_out
                else:
                    prefix = assistant_prefix or ""
                    text = prefix + text_out
                outputs.append(GenerationOutput(prompt=prompt, text=text))
        return outputs
