import gc
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer


@dataclass
class LoadedGemmaMtp:
    model_id: str
    assistant_model_id: str
    model_path: str
    assistant_model_path: str
    device: str
    target_model: Any
    assistant_model: Any
    processor: Any
    num_assistant_tokens: int


_loaded: Optional[LoadedGemmaMtp] = None


def _repo_local_root_dir_name(repo_id: str) -> str:
    return "---".join(repo_id.split("/")[:2])


def _resolve_repo_path(model_root: str, repo_id: str) -> str:
    path = os.path.abspath(os.path.join(model_root, _repo_local_root_dir_name(repo_id)))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Transformers model not found: {path}")
    return path


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _resolve_device(requested_device: Optional[str]) -> str:
    if requested_device and requested_device.startswith("xpu") and _has_xpu():
        return requested_device
    if requested_device == "cpu":
        return "cpu"
    if _has_xpu():
        return "xpu:0"
    return "cpu"


def _dtype_for_device(device: str):
    return torch.bfloat16 if device.startswith("xpu") else torch.float32


def _unload_current() -> None:
    global _loaded
    _loaded = None
    gc.collect()
    if _has_xpu():
        torch.xpu.empty_cache()


def load_model_pair(
    model_id: str,
    assistant_model_id: str,
    model_root: str,
    requested_device: Optional[str],
    num_assistant_tokens: int = 4,
) -> Dict[str, Any]:
    global _loaded

    device = _resolve_device(requested_device)
    model_path = _resolve_repo_path(model_root, model_id)
    assistant_model_path = _resolve_repo_path(model_root, assistant_model_id)

    if (
        _loaded
        and _loaded.model_id == model_id
        and _loaded.assistant_model_id == assistant_model_id
        and _loaded.device == device
        and _loaded.model_path == model_path
        and _loaded.assistant_model_path == assistant_model_path
    ):
        _loaded.num_assistant_tokens = num_assistant_tokens
        _configure_assistant(_loaded.assistant_model, num_assistant_tokens)
        return get_loaded_info()

    _unload_current()

    dtype = _dtype_for_device(device)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_model_path,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    target_model.eval()
    assistant_model.eval()
    _configure_assistant(assistant_model, num_assistant_tokens)

    _loaded = LoadedGemmaMtp(
        model_id=model_id,
        assistant_model_id=assistant_model_id,
        model_path=model_path,
        assistant_model_path=assistant_model_path,
        device=device,
        target_model=target_model,
        assistant_model=assistant_model,
        processor=processor,
        num_assistant_tokens=num_assistant_tokens,
    )
    return get_loaded_info()


def _configure_assistant(assistant_model: Any, num_assistant_tokens: int) -> None:
    assistant_model.generation_config.num_assistant_tokens = num_assistant_tokens
    assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic"


def get_loaded_info() -> Dict[str, Any]:
    if not _loaded:
        return {"loaded": False}
    return {
        "loaded": True,
        "model": _loaded.model_id,
        "assistant_model": _loaded.assistant_model_id,
        "device": _loaded.device,
        "num_assistant_tokens": _loaded.num_assistant_tokens,
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "content": _message_text(message.get("content"))})
    return normalized


def _generation_kwargs(request_json: Dict[str, Any]) -> Dict[str, Any]:
    max_tokens = request_json.get("max_tokens", request_json.get("max_completion_tokens", 256))
    temperature = request_json.get("temperature", 0)
    do_sample = bool(temperature and temperature > 0)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_tokens or 256),
        "do_sample": do_sample,
    }
    if request_json.get("mtp") is not False:
        kwargs["assistant_model"] = _loaded.assistant_model
    if do_sample:
        kwargs["temperature"] = float(temperature)
        if request_json.get("top_p") is not None:
            kwargs["top_p"] = float(request_json["top_p"])
    return kwargs


def _inputs_for_messages(messages: List[Dict[str, Any]]):
    if not _loaded:
        raise RuntimeError("Gemma MTP model is not loaded")
    chat_messages = _normalize_messages(messages)
    input_text = _loaded.processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return _loaded.processor(text=input_text, return_tensors="pt").to(_loaded.device)


def chat_completion(request_json: Dict[str, Any]) -> Dict[str, Any]:
    if not _loaded:
        raise RuntimeError("Gemma MTP model is not loaded")

    messages = request_json.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected messages list")

    start = time.perf_counter()
    inputs = _inputs_for_messages(messages)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        outputs = _loaded.target_model.generate(**inputs, **_generation_kwargs(request_json))
    response_text = _loaded.processor.decode(
        outputs[0][prompt_tokens:],
        skip_special_tokens=True,
    )
    elapsed = time.perf_counter() - start
    completion_tokens = int(outputs.shape[1] - prompt_tokens)

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _loaded.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "mtp": {"enabled": request_json.get("mtp") is not False},
        "timings": {
            "predicted_n": completion_tokens,
            "predicted_ms": elapsed * 1000,
            "predicted_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
        },
    }


def chat_completion_stream(request_json: Dict[str, Any]) -> Iterable[str]:
    if not _loaded:
        raise RuntimeError("Gemma MTP model is not loaded")

    messages = request_json.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected messages list")

    inputs = _inputs_for_messages(messages)
    streamer = TextIteratorStreamer(
        _loaded.processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    kwargs = {**inputs, **_generation_kwargs(request_json), "streamer": streamer}
    completion_id = f"chatcmpl-{uuid.uuid4()}"

    def run_generate():
        with torch.inference_mode():
            _loaded.target_model.generate(**kwargs)

    thread = threading.Thread(target=run_generate)
    thread.start()
    for text in streamer:
        if not text:
            continue
        yield "data: " + _json_chunk(completion_id, text) + "\n\n"
    thread.join()
    yield "data: [DONE]\n\n"


def _json_chunk(completion_id: str, text: str) -> str:
    import json

    return json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _loaded.model_id if _loaded else "",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )
