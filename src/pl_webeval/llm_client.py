import httpx
import time
import json
import base64
import asyncio
import os
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from .data_models import LLMMetrics

def _extract_first_json(text: str) -> Optional[Dict]:
    """Robustly extract the first JSON object from a response text.
    Handles code fences, malformed JSON, and various edge cases.
    Returns a parsed dict on success or None on failure."""
    if not text or not isinstance(text, str):
        return None

    s = text.strip()

    # Strip markdown code fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.startswith("json"):
                s = s[4:].strip()

    # Try direct parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass  # Continue with fixes
    except Exception:
        return None

    # Comprehensive JSON repair function
    def repair_json(json_str: str) -> str:
        """Repair common JSON issues: unescaped quotes, newlines, unbalanced braces."""
        result = []
        in_string = False
        escape_next = False
        brace_stack = 0
        bracket_stack = 0
        i = 0

        while i < len(json_str):
            char = json_str[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                if in_string:
                    if not escape_next:
                        in_string = False
                    result.append(char)
                else:
                    in_string = True
                    result.append(char)
                i += 1
                continue

            if in_string:
                # Handle newlines and tabs in strings
                if char in '\n\r\t':
                    if char == '\n':
                        result.append('\\n')
                    elif char == '\r':
                        result.append('\\r')
                    elif char == '\t':
                        result.append('\\t')
                else:
                    result.append(char)
            else:
                # Track braces and brackets outside strings
                if char == '{':
                    brace_stack += 1
                elif char == '}':
                    brace_stack = max(0, brace_stack - 1)
                elif char == '[':
                    bracket_stack += 1
                elif char == ']':
                    bracket_stack = max(0, bracket_stack - 1)
                result.append(char)

            i += 1

        # Close unterminated string
        if in_string:
            result.append('"')

        # Balance braces and brackets
        if brace_stack > 0:
            result.extend('}' * brace_stack)
        if bracket_stack > 0:
            result.extend(']' * bracket_stack)

        return ''.join(result)

    # Try with comprehensive repair
    try:
        repaired = repair_json(s)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    except Exception:
        return None

    # Try balanced extraction of first complete JSON object
    start = s.find('{')
    if start != -1:
        stack = 0
        in_string = False
        escape_next = False
        for i in range(start, len(s)):
            char = s[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    stack += 1
                elif char == '}':
                    stack -= 1
                    if stack == 0:
                        candidate = s[start:i+1]
                        try:
                            repaired_candidate = repair_json(candidate)
                            return json.loads(repaired_candidate)
                        except json.JSONDecodeError:
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                break

    # Last resort: Python literal replacements and retry
    try:
        s_fixed = re.sub(r'\bTrue\b', 'true', s)
        s_fixed = re.sub(r'\bFalse\b', 'false', s_fixed)
        s_fixed = re.sub(r'\bNone\b', 'null', s_fixed)
        s_fixed = repair_json(s_fixed)
        return json.loads(s_fixed)
    except Exception:
        return None

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
    
    def _optimize_image_data(self, image_data: str, orig_mime: str, max_size: int = 1024 * 1024) -> Tuple[str, str]:
        """Optimize image data by re-encoding and/or resizing if it's too large.
        Returns a tuple of (base64_data, mime_type). Never returns invalid/truncated Base64.
        If optimization cannot be performed safely, returns the original data and mime.
        """
        try:
            decoded = base64.b64decode(image_data, validate=True)
        except Exception as e:
            print(f"    ⚠️ Could not decode screenshot for optimization: {e}; using original")
            return image_data, orig_mime

        if len(decoded) <= max_size:
            return image_data, orig_mime

        print(f"    🖼️ Optimizing large image ({len(decoded)} bytes -> target: {max_size} bytes)")

        # Try to optimize with Pillow if available; otherwise keep original (never truncate arbitrarily)
        try:
            from io import BytesIO
            try:
                from PIL import Image
            except ImportError:
                print("    ⚠️ Pillow not installed; skipping image compression. Using original image.")
                return image_data, orig_mime

            # Load original image
            im = Image.open(BytesIO(decoded))
            # Convert to RGB for JPEG (smaller and widely supported)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            quality = 85
            scale = 1.0
            attempts = 0
            result_bytes = None

            while attempts < 8:
                attempts += 1
                buf = BytesIO()
                save_kwargs = {"format": "JPEG", "optimize": True, "quality": quality}
                # Resize if scaling requested
                if scale < 1.0:
                    new_w = max(64, int(im.width * scale))
                    new_h = max(64, int(im.height * scale))
                    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
                    im_resized.save(buf, **save_kwargs)
                else:
                    im.save(buf, **save_kwargs)

                data = buf.getvalue()
                if len(data) <= max_size or (quality <= 50 and scale <= 0.6):
                    result_bytes = data
                    break

                # First reduce quality down to 50, then start downscaling
                if quality > 50:
                    quality -= 10
                else:
                    scale *= 0.85

            if result_bytes is None:
                # Fallback to last attempt buffer if loop did not break
                result_bytes = data

            optimized_b64 = base64.b64encode(result_bytes).decode("utf-8")
            # We encoded as JPEG
            return optimized_b64, "image/jpeg"

        except Exception as e:
            print(f"    ⚠️ Image optimization pipeline failed: {e}; using original")
            return image_data, orig_mime
    
    def _optimize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Optimize messages by compressing large images."""
        optimized_messages = []
        for message in messages:
            optimized_message = message.copy()
            if isinstance(message.get('content'), list):
                optimized_content = []
                for content_item in message['content']:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get('type') == 'image_url'
                        and isinstance(content_item.get('image_url'), dict)
                    ):
                        image_url = content_item['image_url'].get('url', '')
                        if image_url.startswith('data:image/'):
                            # Extract base64 data and mime
                            try:
                                header, b64_data = image_url.split(',', 1)
                                # header example: "data:image/png;base64"
                                mime_part = header.split(':', 1)[1] if ':' in header else 'image/png;base64'
                                orig_mime = mime_part.split(';', 1)[0] if ';' in mime_part else 'image/png'
                                optimized_b64, optimized_mime = self._optimize_image_data(b64_data, orig_mime)
                                new_header = f"data:{optimized_mime};base64"
                                optimized_item = content_item.copy()
                                optimized_item['image_url'] = {'url': f"{new_header},{optimized_b64}"}
                                optimized_content.append(optimized_item)
                            except Exception as e:
                                print(f"    ⚠️ Could not optimize data URL image: {e}; using original")
                                optimized_content.append(content_item)
                        else:
                            optimized_content.append(content_item)
                    else:
                        optimized_content.append(content_item)
                optimized_message['content'] = optimized_content

            optimized_messages.append(optimized_message)

        return optimized_messages
        
    async def get_completion(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1000,
        analysis_type: str = "",
        temperature: float = 0.7,
        response_format: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Tuple[Dict, LLMMetrics]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Optimize messages to handle large images
        optimized_messages = self._optimize_messages(messages)

        # Model-independent handling - no provider-specific adjustments needed
        effective_messages = optimized_messages
        effective_response_format = response_format

        # Determine if any image inputs are present (for diagnostic logging)
        has_images = False
        try:
            for _m in effective_messages:
                if isinstance(_m.get("content"), list):
                    for _part in _m["content"]:
                        if isinstance(_part, dict) and _part.get("type") == "image_url":
                            has_images = True
                            break
                if has_images:
                    break
        except Exception:
            has_images = False

        payload = {
            "model": model,
            "messages": effective_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if effective_response_format:
            payload["response_format"] = effective_response_format
        
        start_time = time.time()
        response = None
        last_error = None
        
        # Retry logic for empty responses
        for attempt in range(max_retries):
            async with httpx.AsyncClient() as client:
                try:
                    print(f"    🔄 LLM API attempt {attempt + 1}/{max_retries} for {analysis_type} | model={model} | response_format={'yes' if 'response_format' in payload else 'no'} | images={'yes' if has_images else 'no'}")
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=300.0  # Increased timeout for provider variability
                    )
                    response.raise_for_status()
                    
                    # Try to parse response
                    try:
                        result = response.json()
                    except json.JSONDecodeError as je:
                        # Attempt to recover from partial / annotated JSON responses by extracting
                        # the first JSON object substring from the raw response text before giving up.
                        raw_text = ""
                        try:
                            raw_text = response.text or ""
                        except Exception:
                            raw_text = ""
                        parsed = None
                        try:
                            # Heuristic: find first '{' and last '}' and try to parse that slice
                            s = raw_text.find('{')
                            e = raw_text.rfind('}')
                            if s != -1 and e != -1 and e > s:
                                candidate = raw_text[s:e+1]
                                parsed = json.loads(candidate)
                        except Exception:
                            parsed = None
                        if parsed is not None:
                            result = parsed
                        else:
                            # Keep previous retry behavior if we can still retry
                            raw_preview = raw_text[:400]
                            if attempt < max_retries - 1:
                                print(f"    ⚠️ JSON decode error, retrying... ({je}); attempted JSON snippet extraction")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            raise ValueError(f"Non-JSON response from OpenRouter: {je}; status={getattr(response, 'status_code', 'N/A')}; body_preview={raw_preview}") from je
                    
                    # Check for empty content
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not content or not content.strip():
                        # If response_format was requested and we get empty content,
                        # let the calling method handle the retry logic with their own strategy
                        if effective_response_format and attempt == max_retries - 1:
                            print(f"    ⚠️ Empty content with response_format, letting caller handle retry...")
                            # Return the result as-is, let the caller's retry logic handle it
                            break
                        
                        if attempt < max_retries - 1:
                            print(f"    ⚠️ Empty content received, retrying with different parameters...")
                            # Try different strategies to get non-empty content
                            if "response_format" in payload:
                                del payload["response_format"]
                                print(f"    🔧 Removed response_format to improve compatibility")
                            # Increase temperature slightly to encourage more verbose responses
                            if payload.get("temperature", 0.7) < 0.3:
                                payload["temperature"] = 0.3
                                print(f"    🔧 Increased temperature to encourage more verbose responses")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        # Save raw response for debugging only if not using response_format
                        if not effective_response_format:
                            try:
                                raw_response_path = f"raw_empty_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                with open(raw_response_path, 'w', encoding='utf-8') as f:
                                    json.dump(result, f, indent=2)
                                print(f"    📄 Saved raw empty response to {raw_response_path}")
                            except Exception as save_e:
                                print(f"    ⚠️ Could not save raw response: {save_e}")
                            raise ValueError("Empty content received from OpenRouter API after all retries")
                    
                    # Success! Break out of retry loop
                    break
                    
                except httpx.HTTPError as he:
                    last_error = he
                    # If response_format caused incompatibility, retry once without it
                    status = getattr(he.response, "status_code", None) if getattr(he, "response", None) is not None else None

                    # Log response body and parameters for diagnostics on 4xx
                    try:
                        body_text_log = he.response.text if getattr(he, "response", None) is not None else ""
                    except Exception:
                        body_text_log = ""
                    try:
                        print(f"    ❌ HTTP {status} during {analysis_type} | model={model} | attempt={attempt + 1}/{max_retries}")
                        if body_text_log:
                            body_preview_log = body_text_log.replace('\r', ' ').replace('\n', ' ')
                            print(f"    ↪  4xx response body (preview 1000 chars): {body_preview_log[:1000]}")
                        print(f"    ↪  Sent params: response_format={'yes' if 'response_format' in payload else 'no'}, temperature={temperature}, max_tokens={max_tokens}, images={'yes' if has_images else 'no'}")
                    except Exception:
                        pass

                    can_retry_without_format = response_format is not None and status in (400, 422)
                    
                    if can_retry_without_format and "response_format" in payload:
                        print(f"    🔧 Removing response_format due to compatibility issue (status: {status}); will retry without it")
                        payload.pop("response_format", None)
                        # do not return yet; allow next loop iteration
                        continue

                    # If provider complains about inline_data / TYPE_BYTES / Base64 decoding, strip images and retry
                    try:
                        body_text = he.response.text if getattr(he, "response", None) is not None else ""
                    except Exception:
                        body_text = ""

                    # If model is invalid, fail fast with a clear error instead of retrying
                    try:
                        lower_body = (body_text or "").lower()
                        if status in (400, 404) and any(tok in lower_body for tok in [
                            "not a valid model id",
                            "model does not exist",
                            "unknown model",
                            "invalid model"
                        ]):
                            raise RuntimeError(f"Invalid OpenRouter model id '{model}'. Provider response: {body_text[:300]}") from he
                    except Exception:
                        pass
                    if status in (400, 422) and any(tok in body_text for tok in ["inline_data", "TYPE_BYTES", "Base64 decoding failed"]):
                        try:
                            msgs = payload.get("messages", [])
                            modified = False
                            for m in msgs:
                                if isinstance(m.get("content"), list):
                                    filtered = []
                                    for part in m["content"]:
                                        if isinstance(part, dict) and part.get("type") == "image_url":
                                            modified = True
                                            continue
                                        filtered.append(part)
                                    if not filtered:
                                        filtered = [{"type": "text", "text": "Screenshot omitted due to provider constraints. Use textual snapshot only."}]
                                    m["content"] = filtered
                            if modified:
                                print("    🔧 Removed image_url parts from messages due to provider 400 on inline_data/bytes")
                                await asyncio.sleep(2 ** attempt)
                                continue
                        except Exception:
                            pass
                    
                    if attempt < max_retries - 1:
                        print(f"    ⚠️ HTTP error (status: {status}), retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    # Final attempt failed
                    body_preview = ""
                    status_str = status if status is not None else "N/A"
                    try:
                        if he.response is not None:
                            body_preview = he.response.text[:400]
                    except Exception:
                        pass
                    raise RuntimeError(f"HTTP error from OpenRouter after {max_retries} attempts ({type(he).__name__}): status={status_str} body_preview={body_preview}") from he
        
        usage = result.get('usage', {})
        response_time = time.time() - start_time
        metrics = LLMMetrics(
            model=model,
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
            response_time=response_time,
            analysis_type=analysis_type,
            timestamp=datetime.now().isoformat()
        )
        print(f"    ⏱️ LLM response_time={response_time:.2f}s; tokens: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")
        
        return result, metrics