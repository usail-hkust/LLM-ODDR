import requests
import json
from together import Together
# import transformers
import torch
# import vllm
# from transformers import AutoTokenizer


class GPT4Api:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "xxx"
        }
        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}

    def get_completions(self, prompt):
        data = {
            "model": self.model_name,  # or "gpt-4-32k"
            "messages": prompt,
            "temperature": 0.25
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        try:
            results = response.json()['choices'][0]['message']['content']
        except:
            results = ''
        # print(results)
        return results

    def clear_memory(self):
        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}


class TogetherApi:
    def __init__(self, model_path, model_name):
        self.model_name = model_name
        self.model_path = model_path
        self.api_key = 'xxx'
        self.client = Together(api_key=self.api_key)

        if 'Meta-Llama' in model_name:
            self.type = "meta-llama"
        elif 'Qwen' in model_name:
            self.type = "Qwen"

        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}

    def get_completions(self, prompt):
        stream = self.client.chat.completions.create(
            model=f"{self.type}/{self.model_name}",
            messages=prompt,
            max_tokens=32000,
            stream=True,
            temperature=0.25,
        )
        response = ''
        for chunk in stream:
            try:
                response += chunk.choices[0].delta.content
            except:
                pass

        return response

    def clear_memory(self):
        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}


class LlamaAgent:
    def __init__(self, model_path, model_name):
        self.model_name = model_name
        self.model_path = model_path
        self.model_path = f"{self.model_path}/{self.model_name}"
        self.llm_model = vllm.LLM(
            self.model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )
        self.test_generation_kwargs = {
            "top_k": 50,
            "top_p": 0.75,
            "temperature": 0.25,
            "max_tokens": 3200,
        }
        self.sampling_params = vllm.SamplingParams(**self.test_generation_kwargs)

        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}

    def get_completions(self, prompt):
        response = self.llm_model.chat(prompt, sampling_params=self.sampling_params, use_tqdm=False)
        result = response[0].outputs[0].text
        return result

    def clear_memory(self):
        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}


class QwenAgent:
    def __init__(self, model_path, model_name):
        self.model_name = model_name
        self.model_path = model_path
        self.model_path = f"{self.model_path}/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm_model = vllm.LLM(
            self.model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )
        self.test_generation_kwargs = {
            "top_k": 50,
            "top_p": 0.75,
            "temperature": 0.25,
            "max_tokens": 3200,
        }
        self.sampling_params = vllm.SamplingParams(**self.test_generation_kwargs)

        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}

    def get_completions(self, prompt):
        text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        # generate outputs
        response = self.llm_model.generate([text], sampling_params=self.sampling_params, use_tqdm=False)
        result = response[0].outputs[0].text
        return result

    def clear_memory(self):
        self.score_memory = {}
        self.review_memory = {}
        self.dispatch_memory = {}
        self.reposition_memory = {}