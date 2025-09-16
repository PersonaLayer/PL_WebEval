
import asyncio
import json
import csv
import os
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from pathlib import Path
import base64
import re
import argparse

from dotenv import load_dotenv

# Logging (generalized)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("webeval.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Local imports
from .data_models import (
    UXProfile, TestCase, BotDetectionResult, HomepageMetrics,
    ExpertAnalysis, AdaptationScore, VisualComparison, LLMMetrics, HomepageResult
)
from .llm_client import OpenRouterClient
from .report_utils import generate_comprehensive_report_html
from .json_extractor import extract_first_json
from .browser_config import BrowserConfig

# Optional MCP (Playwright)
try:
    from mcp import ClientSession, StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
except ImportError:
    print("MCP library not found. Please ensure it is installed and accessible.")
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


def _fix_b64_padding(b64_str: str) -> str:
    try:
        if not b64_str or not isinstance(b64_str, str):
            return ""
        missing = (-len(b64_str)) % 4
        if missing:
            return b64_str + ("=" * missing)
        return b64_str
    except Exception:
        return b64_str or ""


def _is_valid_base64_png(b64_str: str) -> bool:
    try:
        if not b64_str or not isinstance(b64_str, str):
            return False
        base64.b64decode(b64_str, validate=True)
        return True
    except Exception:
        return False


def _safe_b64_len(b64_str: str) -> int:
    try:
        if not b64_str:
            return 0
        fixed = _fix_b64_padding(b64_str)
        return len(base64.b64decode(fixed, validate=True))
    except Exception:
        return 0


# Load environment
load_dotenv()

# Ensure UTF-8 console (Windows)
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class WebEvaluator:
    """
    PL-WebEval core. Generalized page evaluator orchestrating:
    - Playwright MCP navigation
    - LLM-based bot detection and visual/metrics analysis
    - Persona-conditioned CSS/JS adaptations
    - Artifact and report generation
    """

    def __init__(self, openrouter_api_key: str, output_dir: Path):
        if not OpenRouterClient:
            raise ImportError("OpenRouterClient could not be initialized (missing httpx).")
        self.openrouter = OpenRouterClient(openrouter_api_key)
        self.results: List[HomepageResult] = []
        self.session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self._client_context = None
        self.output_dir = output_dir
        self.current_test_num = 0
        self.all_llm_metrics: List[LLMMetrics] = []
        self.ux_profiles: Dict[str, UXProfile] = self.load_ux_profiles()

        # Use enhanced browser configuration
        self.browser_config = BrowserConfig()
        self.csv_path: Optional[Path] = None
        self.csv_row_map: Dict[str, List[int]] = {}

    def load_ux_profiles(self) -> Dict[str, UXProfile]:
        print("Loading UX profiles from CSV...")
        profiles: Dict[str, UXProfile] = {}
        pkg_root = Path(__file__).resolve().parents[2]  # .../PL_WebEval
        csv_path = (pkg_root / "data" / "ux_profiles.csv")
        if not csv_path.exists():
            csv_path = Path(__file__).parent / "ux_profiles.csv"

        if not csv_path.exists():
            print(f"Warning: UX Profiles CSV not found at {csv_path}. No profiles will be loaded.")
            return profiles

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        profile = UXProfile(
                            id=row.get('ID', ''),
                            name=row.get('Profile Name', ''),
                            description=row.get('Description / Personalization Focus', ''),
                            recommended_actions=row.get('Recommended Actions / Needs', ''),
                            category=row.get('Category', '')
                        )
                        if profile.name:
                            profiles[profile.name] = profile
                            print(f"Loaded UX profile: {profile.name} (ID: {profile.id})")
                    except KeyError as e:
                        print(f"Skipping row (missing key {e}): {row}")
        except Exception as e:
            print(f"Error loading UX profiles: {e}")
        return profiles


    def _normalize_homepage_metrics_dict(self, d: Dict) -> Dict:
        """Coerce HomepageMetrics fields to correct types and ensure presence."""
        try:
            d = d or {}
        except Exception:
            d = {}
        def _to_int(v):
            try:
                if isinstance(v, bool):
                    return int(v)
                return int(float(v))
            except Exception:
                return 0
        def _to_float(v):
            try:
                if isinstance(v, bool):
                    return float(int(v))
                return float(v)
            except Exception:
                return 0.0
        out = {
            "elements_count": _to_int(d.get("elements_count", 0)),
            "interactive_elements_count": _to_int(d.get("interactive_elements_count", 0)),
            "accessibility_score": _to_float(d.get("accessibility_score", 0.0)),
            "visual_complexity_score": _to_float(d.get("visual_complexity_score", 0.0)),
            "color_contrast_issues": _to_int(d.get("color_contrast_issues", 0)),
            "text_readability_score": _to_float(d.get("text_readability_score", 0.0)),
            "adaptation_effectiveness_score": _to_float(d.get("adaptation_effectiveness_score", 0.0)),
        }
        # Optional field preserved as string when present
        if "analysis_reasoning" in d:
            try:
                out["analysis_reasoning"] = str(d.get("analysis_reasoning") or "")
            except Exception:
                out["analysis_reasoning"] = ""
        return out

    def _save_raw_llm(self, filename_prefix: str, content: str) -> Optional[str]:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path = self.output_dir / fname
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content or "")
            print(f"    Saved raw LLM content to {path}")
            return str(path)
        except Exception as e:
            print(f"    Failed to save raw LLM content: {e}")
            return None

    async def connect_to_playwright_mcp(self):
        if not StdioServerParameters or not stdio_client or not ClientSession:
            msg = "Playwright MCP components not available. Cannot connect."
            logger.error(msg)
            print(msg)
            return

        # Resolve Playwright MCP CLI path robustly (ENV -> local vendor near this file -> project root fallbacks)
        env_cli = os.getenv("PLAYWRIGHT_MCP_CLI_PATH")
        candidates = []
        if env_cli:
            candidates.append(Path(env_cli))

        script_dir = Path(__file__).resolve().parent  # .../PL_WebEval/src/pl_webeval
        candidates.append(script_dir / "playwright-mcp-main" / "cli.js")

        pkg_root = Path(__file__).resolve().parents[2]  # .../PL_WebEval
        candidates.append(pkg_root / "playwright-mcp-main" / "cli.js")
        # Also try an explicit src/pl_webeval vendor path relative to project root
        candidates.append(pkg_root / "src" / "pl_webeval" / "playwright-mcp-main" / "cli.js")

        cli_path = None
        tried_paths = []
        for p in candidates:
            tried_paths.append(str(p))
            try:
                if p.exists():
                    cli_path = p
                    break
            except Exception:
                continue

        if not cli_path:
            print("Playwright MCP CLI not found in any expected location:")
            for tp in tried_paths:
                print(f" - {tp}")
            print("Set PLAYWRIGHT_MCP_CLI_PATH to the full path of playwright-mcp-main/cli.js or place the vendor folder next to evaluator.py.")
            return

        print(f"Using Playwright MCP CLI: {cli_path}")

        # Build MCP CLI arguments using only supported flags.
        # Do NOT pass raw Chromium flags here — the MCP CLI will exit if it sees unknown args.
        # Default to headed browser (more human-like, reduces bot detection like on coupang.com)
        use_headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"

        # Prefer browser choice from env, then try sensible fallbacks on Windows
        preferred = os.getenv("PLAYWRIGHT_BROWSER", "msedge").lower().strip()
        candidates = [b for b in [preferred, "msedge", "chrome", "firefox", "webkit"] if b]

        connect_errors = []
        for browser_choice in dict.fromkeys(candidates):  # preserve order, de-dup
            try:
                arguments = [
                    str(cli_path.absolute()),
                    "--viewport-size", "1920,1080",
                    "--isolated",
                    "--browser", browser_choice,
                    "--no-sandbox",
                    "--ignore-https-errors",  # Ignore certificate errors
                ]
                if use_headless:
                    arguments.append("--headless")

                print(f"Trying Playwright MCP with browser={browser_choice}, headless={use_headless}")
                server_params = StdioServerParameters(command="node", args=arguments, env=None)

                # Ensure any previous partial context is cleaned up before retrying
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self.session = None
                if self._client_context:
                    try:
                        await self._client_context.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._client_context = None

                self._client_context = stdio_client(server_params)
                self.read_stream, self.write_stream = await self._client_context.__aenter__()
                self.session = ClientSession(self.read_stream, self.write_stream)
                await self.session.__aenter__()
                await self.session.initialize()
                print(f"Connected to Playwright MCP with browser={browser_choice}")
                break
            except Exception as e:
                connect_errors.append(f"{browser_choice}: {e}")
                # Clean up before next attempt
                try:
                    if self.session:
                        await self.session.__aexit__(None, None, None)
                except Exception:
                    pass
                self.session = None
                try:
                    if self._client_context:
                        await self._client_context.__aexit__(None, None, None)
                except Exception:
                    pass
                self._client_context = None
                await asyncio.sleep(0.2)
                continue

        if not self.session:
            print("Failed to start Playwright MCP after trying browsers:")
            for err in connect_errors:
                print(f"  - {err}")
            return

        # Set random user agent for bot evasion
        user_agent = BrowserConfig.get_random_user_agent()
        try:
            await self.session.call_tool("browser_set_user_agent", {"userAgent": user_agent})
            print(f"Set user agent: {user_agent[:50]}...")
        except Exception as e:
            print(f"Could not set user agent: {e}, continuing...")
        
        # Inject stealth scripts to avoid detection
        print("Injecting stealth scripts for bot evasion...")
        stealth_scripts = BrowserConfig.get_stealth_scripts()
        for script in stealth_scripts:
            try:
                await self.session.call_tool("browser_inject_js", {"javascript": script})
            except Exception as e:
                logger.debug(f"Could not inject stealth script: {e}")
        
        # Set additional browser preferences for compatibility
        try:
            # Set extra HTTP headers
            headers = BrowserConfig.get_stealth_headers()
            await self.session.call_tool("browser_set_extra_headers", {"headers": headers})
        except:
            pass
        
        try:
            # Enable JavaScript (crucial for many sites)
            await self.session.call_tool("browser_enable_javascript", {"enabled": True})
        except:
            pass
        
        try:
            # Accept cookies automatically
            await self.session.call_tool("browser_accept_cookies", {"accept": True})
        except:
            pass
        
        try:
            # Set viewport to common desktop size
            await self.session.call_tool("browser_set_viewport", {"width": 1920, "height": 1080})
        except:
            pass
            
        print("Connected to Playwright MCP server with maximum stealth and compatibility")

    async def disconnect(self):
        if self.session:
            try:
                await self.session.call_tool("browser_close", {})
            except Exception as e:
                print(f"Error during browser_close: {e}")
            try:
                await self.session.__aexit__(None, None, None)
                self.session = None
            except Exception as e:
                print(f"Error closing session: {e}")

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
                self._client_context = None
            except Exception as e:
                print(f"Error closing client context: {e}")
        await asyncio.sleep(0.1)

    def save_screenshot(self, screenshot_data: str, filepath: Path) -> int:
        if screenshot_data:
            try:
                fixed_b64 = _fix_b64_padding(screenshot_data)
                image_data = base64.b64decode(fixed_b64, validate=True)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                print(f"    Saved screenshot: {filepath.name}")
                return len(image_data)
            except Exception as e:
                print(f"    Failed to save screenshot {filepath.name}: {e}")
                return 0
        return 0

    def save_html_snapshot(self, html_content: str, filepath: Path) -> int:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"    Saved HTML snapshot: {filepath.name}")
            return len(html_content.encode('utf-8'))
        except Exception as e:
            print(f"    Failed to save HTML snapshot {filepath.name}: {e}")
            return 0

    async def take_screenshot(self) -> str:
        if not self.session:
            return ""
        max_retries = 3
        full_page_pref = (os.getenv("PLAYWRIGHT_FULLPAGE", "false").lower() == "true")
        for attempt in range(max_retries):
            try:
                # Try full-page first if requested; fall back to viewport if unsupported
                payload = {"raw": True}
                if full_page_pref:
                    payload["fullPage"] = True
                try:
                    result = await self.session.call_tool("browser_take_screenshot", payload)
                except Exception:
                    # Retry without fullPage if the tool doesn't support it
                    if "fullPage" in payload:
                        payload.pop("fullPage", None)
                        result = await self.session.call_tool("browser_take_screenshot", payload)
                    else:
                        raise

                if result and hasattr(result, 'content') and result.content:
                    for content_item in result.content:
                        if hasattr(content_item, 'type') and content_item.type == 'image' and hasattr(content_item, 'data'):
                            b64 = _fix_b64_padding(content_item.data)
                            if _is_valid_base64_png(b64):
                                return b64
                return ""
            except Exception as e:
                print(f"    Screenshot attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    print("    All screenshot attempts failed")
                    return ""
        return ""

    async def get_page_snapshot(self) -> str:
        if not self.session:
            return ""
        try:
            result = await self.session.call_tool("browser_snapshot", {})
            if result and hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        return content_item.text
            return ""
        except Exception as e:
            print(f"    Get page snapshot failed: {e}")
            return ""

    def update_csv_status(self, website: str, status: str):
        try:
            if not self.csv_path:
                return
            csv_path = self.csv_path
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            updated = False
            if 'Status' not in fieldnames:
                fieldnames.append('Status'); updated = True
            if 'TestCaseID' not in fieldnames:
                fieldnames.append('TestCaseID'); updated = True

            target_idx = None
            if website in self.csv_row_map and self.csv_row_map[website]:
                candidate = self.csv_row_map[website].pop(0)
                li = candidate - 1
                if 0 <= li < len(rows):
                    target_idx = li
            if target_idx is None:
                for i, r in enumerate(rows):
                    if (r.get('Website', '') == website) and (r.get('Status', '').lower() != 'completed'):
                        target_idx = i; break
                if target_idx is None:
                    for i, r in enumerate(rows):
                        if r.get('Website', '') == website:
                            target_idx = i; break

            if target_idx is not None:
                rows[target_idx]['Status'] = status
                updated = True

            if updated:
                for r in rows:
                    r.setdefault('Status', '')
                    r.setdefault('TestCaseID', r.get('TestCaseID', ''))
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            print(f"    ⚠️ Failed to update CSV status for website '{website}': {e}")

    async def llm_analyze_bot_detection(self, screenshot_data: str, html_snapshot: str, model: str = "openai/gpt-4o-mini") -> BotDetectionResult:
        messages = [
            {
                "role": "system",
                "content": """You are an expert web automation specialist analyzing pages for bot detection mechanisms.

Provide detailed analysis including your reasoning process, technical observations, and strategic recommendations.

Analyze the provided screenshot and HTML to determine:
1. Is this a bot detection/blocking page?
2. What type of blocking mechanism is present?
3. Where should the automation click to proceed?
4. Detailed reasoning for your analysis

Common bot detection patterns:
- CAPTCHA challenges (reCAPTCHA, hCaptcha, etc.)
- Cloudflare "Checking your browser" pages
- "Access Denied" or "Blocked" messages
- "Prove you are human" verification
- Rate limiting messages
- JavaScript challenges
- Browser fingerprinting checks

Respond in JSON format:
{
    "is_bot_detected": true/false,
    "confidence_score": 0.0-1.0,
    "detection_type": "captcha|cloudflare|access_denied|verification|rate_limit|js_challenge|fingerprint|none",
    "recommended_action": "Detailed description of what to do",
    "click_coordinates": [x, y] or null,
    "alternative_strategy": "Alternative approach if clicking fails",
    "llm_reasoning": "Detailed step-by-step reasoning for the analysis",
    "technical_observations": "Technical details observed in HTML/visual elements",
    "confidence_factors": "What factors contributed to the confidence score"
}"""
            }
        ]

        content_items = []
        if screenshot_data and _is_valid_base64_png(screenshot_data):
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}
            })
        content_items.append({
            "type": "text",
            "text": f"Analyze this page for bot detection mechanisms with detailed reasoning.\n\nHTML Snapshot (first 3000 chars):\n{html_snapshot[:3000]}..."
        })
        messages.append({"role": "user", "content": content_items})

        start_time = time.time()
        content = ""
        try:
            response, metrics = await self.openrouter.get_completion(
                model, messages,
                max_tokens=800,
                analysis_type="bot_detection",
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "BotDetectionResult",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "is_bot_detected": {"type": "boolean"},
                                "confidence_score": {"type": "number"},
                                "detection_type": {"type": "string"},
                                "recommended_action": {"type": "string"},
                                "click_coordinates": {
                                    "type": ["array", "null"],
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                "alternative_strategy": {"type": "string"},
                                "llm_reasoning": {"type": "string"},
                                "technical_observations": {"type": "string"},
                                "confidence_factors": {"type": "string"}
                            },
                            "required": ["is_bot_detected", "confidence_score", "detection_type"]
                        }
                    }
                }
            )
            content = (response.get('choices', [{}])[0].get('message', {}).get('content', '') or '')
            parsed = extract_first_json(content, context='bot_detection')
            if parsed is None:
                raw_path = self._save_raw_llm("raw_bot_detection_response", content)
                raise ValueError(f"Non-JSON response from model. Saved preview to {raw_path or 'N/A'}")

            click_coords = None
            if parsed.get('click_coordinates') and isinstance(parsed['click_coordinates'], list) and len(parsed['click_coordinates']) == 2:
                click_coords = tuple(map(int, parsed['click_coordinates']))

            try:
                self.all_llm_metrics.append(metrics)
                print(f"    ⏱️ bot_detection LLM time={metrics.response_time:.2f}s; tokens: prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens}")
            except Exception as _me:
                print(f"    ⚠️ Failed to record bot_detection metrics: {_me}")

            return BotDetectionResult(
                is_bot_detected=parsed.get('is_bot_detected', False),
                confidence_score=float(parsed.get('confidence_score', 0.0)),
                detection_type=str(parsed.get('detection_type', 'none')),
                recommended_action=str(parsed.get('recommended_action', '')),
                click_coordinates=click_coords,
                alternative_strategy=str(parsed.get('alternative_strategy', '')),
                llm_reasoning=str(parsed.get('llm_reasoning', '')),
                technical_observations=str(parsed.get('technical_observations', '')),
                confidence_factors=str(parsed.get('confidence_factors', '')),
                analysis_timestamp=metrics.timestamp
            )
        except Exception as e:
            print(f"    Failed to analyze bot detection with LLM: {e}")
            try:
                raw_path = self._save_raw_llm("raw_bot_detection_response", content if 'content' in locals() else "")
            except Exception:
                raw_path = None
            try:
                elapsed = time.time() - start_time
            except Exception:
                elapsed = 0.0
            error_metric = LLMMetrics(model=model, analysis_type="bot_detection", timestamp=datetime.now().isoformat(), response_time=elapsed)
            try:
                self.all_llm_metrics.append(error_metric)
            except Exception:
                pass
            print(f"    ⏱️ bot_detection error LLM time={elapsed:.2f}s; returning fallback result")
            return BotDetectionResult(
                is_bot_detected=False, confidence_score=0.0, detection_type='error',
                recommended_action='Analysis failed',
                llm_reasoning=f"Analysis failed: {str(e)}",
                technical_observations=f"Raw saved: {raw_path or 'N/A'}",
                confidence_factors="Analysis error",
                analysis_timestamp=error_metric.timestamp
            )

    async def handle_bot_detection_with_llm(self, bot_result: BotDetectionResult, current_test_llm_metrics: List[LLMMetrics]) -> bool:
        if not self.session:
            return False
        if not bot_result.is_bot_detected:
            return True

        print(f"    🤖 Bot detection found: {bot_result.detection_type} (confidence: {bot_result.confidence_score:.2f})")
        print(f"    📋 Recommended action: {bot_result.recommended_action}")
        print(f"    🧠 LLM reasoning: {bot_result.llm_reasoning[:100]}...")

        if bot_result.click_coordinates:
            try:
                x, y = bot_result.click_coordinates
                print(f"    🖱️ Attempting to click at coordinates ({x}, {y})")
                await self.session.call_tool("browser_click", {"coordinate": f"{x},{y}"})
                await self.session.call_tool("browser_wait_for", {"time": 3})

                new_screenshot = await self.take_screenshot()
                new_snapshot = await self.get_page_snapshot()
                # Need to pass model parameter - get it from the first bot detection metric
                model_for_reanalysis = next((m.model for m in current_test_llm_metrics if m.analysis_type == "bot_detection"), "openai/gpt-4o-mini")
                new_bot_result = await self.llm_analyze_bot_detection(new_screenshot, new_snapshot, model_for_reanalysis)

                # Attach the matching metrics entry (best-effort)
                try:
                    matched = next(m for m in self.all_llm_metrics if m.analysis_type == "bot_detection" and m.timestamp == new_bot_result.analysis_timestamp)
                    current_test_llm_metrics.append(matched)
                except Exception:
                    pass

                if not new_bot_result.is_bot_detected:
                    print("    ✅ Successfully bypassed bot detection!")
                    return True
                else:
                    print(f"    ⚠️ Bot detection still present after click (Type: {new_bot_result.detection_type}, Confidence: {new_bot_result.confidence_score:.2f})")
            except Exception as e:
                print(f"    ❌ Failed to click or re-analyze: {e}")

        if bot_result.alternative_strategy:
            print(f"    💡 Alternative strategy suggested: {bot_result.alternative_strategy}")
            # Future: implement refresh/retry heuristics
        return False

    async def navigate_to_url(self, url: str, current_test_llm_metrics: List[LLMMetrics], model: str = "openai/gpt-4o-mini") -> Tuple[float, bool, Optional[BotDetectionResult]]:
        if not self.session:
            return 0.0, True, None
        start_time = time.time()
        bot_detected_finally = False
        final_bot_result = None
        try:
            await asyncio.sleep(random.uniform(0.5, 1.5))
            print(f"    Navigating to: {url}")

            # Apply domain-specific stealth overrides (e.g., coupang.com)
            try:
                overrides = BrowserConfig.get_domain_overrides(url)
                if overrides:
                    ua = overrides.get("user_agent")
                    if ua:
                        await self.session.call_tool("browser_set_user_agent", {"userAgent": ua})
                        print(f"    Applied domain UA override")
                    headers = overrides.get("headers")
                    if headers:
                        await self.session.call_tool("browser_set_extra_headers", {"headers": headers})
                        print(f"    Applied domain header overrides")
                    viewport = overrides.get("viewport")
                    if viewport and isinstance(viewport, dict):
                        vw = int(viewport.get("width", 1920)); vh = int(viewport.get("height", 1080))
                        await self.session.call_tool("browser_set_viewport", {"width": vw, "height": vh})
                        print(f"    Applied viewport override: {vw}x{vh}")
                    # Inject locale/timezone spoofing JS if provided
                    for js in (overrides.get("js_snippets") or []):
                        try:
                            await self.session.call_tool("browser_inject_js", {"javascript": js})
                        except Exception:
                            pass
                    # Short human-like delay
                    await self.session.call_tool("browser_wait_for", {"time": 1})

                # Domain-specific URL rewrite for mobile preference (e.g., Coupang prefers mobile site)
                try:
                    import re as _re
                    # If using coupang.com desktop, prefer mobile subdomain for better compatibility
                    if "coupang.com" in url and "m.coupang.com" not in url:
                        new_url = _re.sub(r"://(www\\.)?coupang\\.com", "://m.coupang.com", url)
                        if new_url != url:
                            print(f"    Rewriting to mobile domain for Coupang: {new_url}")
                            url = new_url

                    # Prefer full-page screenshots on aggressive bot sites by default (can be overridden via env)
                    os.environ["PLAYWRIGHT_FULLPAGE"] = os.getenv("PLAYWRIGHT_FULLPAGE", "true")
                except Exception:
                    pass
            except Exception as _ovr_e:
                print(f"    ⚠️ Domain override apply failed: {_ovr_e}")

            # Try navigation with error handling
            try:
                await self.session.call_tool("browser_navigate", {"url": url})
                # Humanization: allow JS verification pages to settle, do small scroll to look human
                await self.session.call_tool("browser_wait_for", {"time": 4})
                try:
                    await self.session.call_tool("browser_inject_js", {"javascript": "try{window.scrollBy({top: 300, behavior:'smooth'});}catch(e){}"})
                except Exception:
                    pass
                await self.session.call_tool("browser_wait_for", {"time": 3})
            except Exception as nav_error:
                print(f"    ⚠️ Navigation error: {nav_error}")
                # Continue to check what actually loaded

            screenshot = await self.take_screenshot()
            snapshot = await self.get_page_snapshot()
            
            # First differentiate between error pages and bot detection
            is_error = BrowserConfig.is_error_page(snapshot, "")
            is_bot_detect = BrowserConfig.is_bot_detection_page(snapshot, "")
            
            # If both indicators present, bot detection takes precedence (site exists but blocking us)
            if is_bot_detect and not is_error:
                print(f"    🤖 Bot detection indicators found, will analyze with LLM...")
            elif is_error and not is_bot_detect:
                print(f"    ❌ Network/DNS error detected (website unreachable)")
                # This is a true network error, not bot detection
                error_bot_result = BotDetectionResult(
                    is_bot_detected=True,
                    confidence_score=1.0,
                    detection_type="unreachable_website",
                    recommended_action="Website is unreachable due to network/DNS error. Verify URL is correct.",
                    llm_reasoning="Network error page detected - website cannot be reached",
                    technical_observations="Page contains network error indicators (ERR_NAME_NOT_RESOLVED, DNS errors, etc.)",
                    analysis_timestamp=datetime.now().isoformat()
                )
                return time.time() - start_time, True, error_bot_result
            elif is_error and is_bot_detect:
                # Both present - likely bot detection causing the "error"
                print(f"    ⚠️ Possible bot detection masquerading as error page, analyzing...")
            
            # Now do the LLM analysis for bot detection
            initial_bot_result = await self.llm_analyze_bot_detection(screenshot, snapshot, model)
            final_bot_result = initial_bot_result
            
            # Cross-check LLM results with our detection
            if not initial_bot_result.is_bot_detected:
                # LLM thinks it's fine, but double-check
                if BrowserConfig.is_error_page(snapshot, "") and not BrowserConfig.is_bot_detection_page(snapshot, ""):
                    print(f"    ⚠️ LLM missed network error detection, overriding...")
                    final_bot_result = BotDetectionResult(
                        is_bot_detected=True,
                        confidence_score=1.0,
                        detection_type="unreachable_website",
                        recommended_action="Website unreachable (network/DNS error)",
                        llm_reasoning="Network error indicators found despite LLM analysis",
                        technical_observations=snapshot[:500],
                        analysis_timestamp=datetime.now().isoformat()
                    )
                    bot_detected_finally = True
                elif BrowserConfig.is_bot_detection_page(snapshot, ""):
                    print(f"    ⚠️ LLM missed bot detection, re-analyzing...")
                    # Force re-analysis with stronger prompt
                    initial_bot_result = await self.llm_analyze_bot_detection(screenshot, snapshot, model)
                    final_bot_result = initial_bot_result
                    if initial_bot_result.is_bot_detected:
                        success = await self.handle_bot_detection_with_llm(initial_bot_result, current_test_llm_metrics)
                        bot_detected_finally = not success
            elif initial_bot_result.is_bot_detected and initial_bot_result.confidence_score > 0.7:
                print(f"    🤖 LLM detected bot blocking with {initial_bot_result.confidence_score:.1%} confidence")
                success = await self.handle_bot_detection_with_llm(initial_bot_result, current_test_llm_metrics)
                bot_detected_finally = not success
                if not success:
                    print(f"    ❌ Could not bypass bot detection for {url}")
                else:
                    print(f"    ✅ Bot detection handled for {url}")
                    final_bot_result = await self.llm_analyze_bot_detection(
                        await self.take_screenshot(),
                        await self.get_page_snapshot(),
                        model
                    )
            else:
                print(f"    ✅ LLM analysis: No significant bot detection (Confidence: {initial_bot_result.confidence_score:.1%}) for {url}")
                bot_detected_finally = False
            
            await self.session.call_tool("browser_wait_for", {"time": 2})
            load_time = time.time() - start_time
            return load_time, bot_detected_finally, final_bot_result
            
        except Exception as e:
            print(f"    Navigation to {url} failed: {e}")
            load_time = time.time() - start_time
            error_bot_result = BotDetectionResult(
                is_bot_detected=True,
                confidence_score=1.0,
                detection_type="navigation_error",
                recommended_action="Check URL or network connection.",
                llm_reasoning=f"Navigation failed: {e}",
                technical_observations=str(e),
                analysis_timestamp=datetime.now().isoformat()
            )
            return load_time, True, error_bot_result

    async def analyze_css_js_with_llm(self, html_snapshot: str, ux_profile_name: str, model: str = "openai/gpt-4o-mini") -> Tuple[Dict, Optional[LLMMetrics]]:
        messages = [
            {
                "role": "system",
                "content": """You are an expert frontend developer analyzing CSS and JavaScript code for accessibility and UX improvements.

Analyze the HTML snapshot and provide detailed technical analysis of:
1. CSS patterns and their impact on accessibility
2. JavaScript functionality and user interaction patterns
3. Code quality and best practices
4. Specific improvements for the given UX profile

Respond in JSON format:
{
  "css_analysis": {
    "patterns_found": ["list of CSS patterns"],
    "accessibility_impact": "detailed analysis",
    "color_contrast_issues": ["list of issues"],
    "typography_analysis": "font choices and readability",
    "responsive_design": "mobile/desktop compatibility",
    "recommendations": ["specific CSS improvements"]
  },
  "js_analysis": {
    "interaction_patterns": ["list of JS interactions"],
    "accessibility_features": ["ARIA, keyboard navigation, etc"],
    "performance_impact": "analysis of JS performance",
    "event_handling": "how events are managed",
    "recommendations": ["specific JS improvements"]
  },
  "code_quality": {
    "score": "0-100",
    "issues": ["list of code quality issues"],
    "best_practices_followed": ["list of good practices"],
    "technical_debt": "assessment of technical debt"
  }
}"""
            },
            {"role": "user", "content": f"Analyze this HTML snapshot for {ux_profile_name} profile:\n\n{html_snapshot[:5000]}..."}
        ]
        content = ""
        try:
            response, metrics = await self.openrouter.get_completion(
                model, messages,
                max_tokens=1200,
                analysis_type="css_js_analysis",
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CSSJSAnalysis",
                        "schema": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "css_analysis": {"type": "object"},
                                "js_analysis": {"type": "object"},
                                "code_quality": {"type": "object", "properties": {"score": {"type": ["integer", "string"]}}}
                            },
                            "required": ["css_analysis", "js_analysis", "code_quality"]
                        }
                    }
                }
            )
            content = (response.get('choices', [{}])[0].get('message', {}).get('content', '') or '')
            analysis_result = extract_first_json(content, context='css_js_analysis')
            if analysis_result is None:
                raw_path = self._save_raw_llm("raw_css_js_analysis_response", content)
                raise ValueError(f"Non-JSON response from model. Saved preview to {raw_path or 'N/A'}")
            if 'code_quality' in analysis_result and 'score' in analysis_result['code_quality']:
                try:
                    analysis_result['code_quality']['score'] = int(analysis_result['code_quality']['score'])
                except ValueError:
                    analysis_result['code_quality']['score'] = 0
            return analysis_result, metrics
        except Exception as e:
            print(f"    Failed to analyze CSS/JS with LLM: {e}")
            try:
                raw_path = self._save_raw_llm("raw_css_js_analysis_response", content if 'content' in locals() else "")
            except Exception:
                raw_path = None
            error_metric = LLMMetrics(model=model, analysis_type="css_js_analysis", timestamp=datetime.now().isoformat())
            return {
                "css_analysis": {"error": str(e)},
                "js_analysis": {"error": str(e)},
                "code_quality": {"score": 0, "error": str(e)},
                "raw_preview": (content[:400] if 'content' in locals() else ""),
                "raw_saved_path": raw_path or ""
            }, error_metric

    async def analyze_homepage_metrics(self, snapshot: str, screenshot_data: str, model: str = "openai/gpt-4o-mini") -> Tuple[HomepageMetrics, Optional[LLMMetrics]]:
        messages = [
            {
                "role": "system",
                "content": """You are a scientific web accessibility and UX metrics analyzer. Analyze the provided homepage snapshot and screenshot to calculate objective metrics.

Calculate these metrics based on the accessibility snapshot and visual analysis:

1. elements_count: Total number of interactive and non-interactive elements
2. interactive_elements_count: Number of buttons, links, inputs, etc.
3. accessibility_score: 0-100 score based on WCAG compliance indicators
4. visual_complexity_score: 0-100 score based on layout complexity, number of sections, visual hierarchy
5. color_contrast_issues: Number of potential contrast problems detected
6. text_readability_score: 0-100 score based on text density, font sizes, spacing
7. adaptation_effectiveness_score: 0-100 score for how well the page could be adapted for accessibility

Respond in JSON format:
{
    "elements_count": 0,
    "interactive_elements_count": 0,
    "accessibility_score": 0.0,
    "visual_complexity_score": 0.0,
    "color_contrast_issues": 0,
    "text_readability_score": 0.0,
    "adaptation_effectiveness_score": 0.0,
    "analysis_reasoning": "Detailed explanation of the scoring"
}"""
            }
        ]
        user_content_list = []
        if screenshot_data and _is_valid_base64_png(screenshot_data):
            user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}})
        user_content_list.append({"type": "text", "text": f"Analyze this homepage for scientific UX metrics.\n\nAccessibility Snapshot:\n{snapshot[:4000]}..."})
        messages.append({"role": "user", "content": user_content_list})

        start_time = time.time()
        content = ""
        try:
            response, llm_metric_data = await self.openrouter.get_completion(
                model, messages,
                max_tokens=1000,
                analysis_type="metrics_analysis",
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "HomepageMetrics",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "elements_count": {"type": "integer"},
                                "interactive_elements_count": {"type": "integer"},
                                "accessibility_score": {"type": "number"},
                                "visual_complexity_score": {"type": "number"},
                                "color_contrast_issues": {"type": "integer"},
                                "text_readability_score": {"type": "number"},
                                "adaptation_effectiveness_score": {"type": "number"},
                                "analysis_reasoning": {"type": "string"}
                            },
                            "required": [
                                "elements_count", "interactive_elements_count", "accessibility_score",
                                "visual_complexity_score", "color_contrast_issues", "text_readability_score",
                                "adaptation_effectiveness_score"
                            ]
                        }
                    }
                }
            )
            try:
                print(f"    ⏱️ metrics_analysis LLM time={llm_metric_data.response_time:.2f}s; tokens: prompt={llm_metric_data.prompt_tokens}, completion={llm_metric_data.completion_tokens}")
            except Exception:
                pass

            msg = (response.get('choices', [{}])[0].get('message', {}) or {})
            parsed_direct = msg.get('parsed')
            if parsed_direct is not None:
                result_json = parsed_direct
            else:
                content = (msg.get('content', '') or '')
                result_json = extract_first_json(content, context='metrics_analysis')
            try:
                path_used = 'parsed_direct' if parsed_direct is not None else 'content_extractor'
            except Exception:
                path_used = 'content_extractor'
            if result_json is None:
                # Retry once with json_object and ultra-strict JSON-only instruction to force valid JSON
                retry_messages = [
                    {
                        "role": "system",
                        "content": """Return ONLY a valid JSON object with these exact keys and types:
{
  "elements_count": integer,
  "interactive_elements_count": integer,
  "accessibility_score": number,
  "visual_complexity_score": number,
  "color_contrast_issues": integer,
  "text_readability_score": number,
  "adaptation_effectiveness_score": number,
  "analysis_reasoning": string
}
Rules:
- Output ONLY the JSON object, nothing else.
- Escape ALL internal double quotes in strings (e.g., \"), no unescaped quotes.
- No trailing commas, no comments, no code fences.
- Do not include newlines inside strings unless escaped as \\n."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Re-evaluate the homepage metrics using the same inputs.\n\nAccessibility Snapshot (first 4000 chars):\n{snapshot[:4000]}..."},
                            *(([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}}] if (screenshot_data and _is_valid_base64_png(screenshot_data)) else []))
                        ]
                    }
                ]
                try:
                    response2, llm_metric_data2 = await self.openrouter.get_completion(
                        model, retry_messages,
                        max_tokens=1000,
                        analysis_type="metrics_analysis_retry",
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    msg2 = (response2.get('choices', [{}])[0].get('message', {}) or {})
                    if msg2.get('parsed') is not None:
                        result_json = msg2.get('parsed')
                        path_used = 'retry_parsed'
                    else:
                        content2 = (msg2.get('content', '') or '')
                        result_json = extract_first_json(content2, context='metrics_analysis_retry')
                        path_used = 'retry_extractor'
                    if result_json is not None:
                        llm_metric_data = llm_metric_data2  # prefer retry metrics if successful
                except Exception:
                    result_json = None

                if result_json is None:
                    raw_path = self._save_raw_llm("raw_metrics_analysis_response", content)
                    raise ValueError(f"Non-JSON response from model. Saved preview to {raw_path or 'N/A'}")
            
            # Normalize and log parsing path
            result_json = self._normalize_homepage_metrics_dict(result_json)
            try:
                logger.info(f"metrics_analysis: using {path_used}; keys={list((result_json or {}).keys())[:8]}")
            except Exception:
                pass

            metrics_obj = HomepageMetrics(
                elements_count=int(result_json.get('elements_count', 0)),
                interactive_elements_count=int(result_json.get('interactive_elements_count', 0)),
                accessibility_score=float(result_json.get('accessibility_score', 0.0)),
                visual_complexity_score=float(result_json.get('visual_complexity_score', 0.0)),
                color_contrast_issues=int(result_json.get('color_contrast_issues', 0)),
                text_readability_score=float(result_json.get('text_readability_score', 0.0)),
                adaptation_effectiveness_score=float(result_json.get('adaptation_effectiveness_score', 0.0))
            )
            return metrics_obj, llm_metric_data
        except Exception as e:
            print(f"    Failed to analyze page metrics: {e}")
            try:
                raw_path = self._save_raw_llm("raw_metrics_analysis_response", content if 'content' in locals() else "")
            except Exception:
                raw_path = None
            try:
                elapsed = time.time() - start_time
            except Exception:
                elapsed = 0.0
            error_metric = LLMMetrics(model=model, analysis_type="metrics_analysis", timestamp=datetime.now().isoformat(), response_time=elapsed)
            print(f"    ⏱️ metrics_analysis error LLM time={elapsed:.2f}s; returning fallback metrics")
            return HomepageMetrics(), error_metric

    async def generate_ux_adaptations(self, ux_profile: UXProfile, website: str, model: str = "openai/gpt-4o-mini", screenshot_data: Optional[str] = None, baseline_snapshot: str = "") -> Tuple[str, str, str, str, Optional[LLMMetrics]]:
        user_content_text = f"""
UX Profile: {ux_profile.name}
Category: {ux_profile.category}
Description: {ux_profile.description}
Recommended Actions: {ux_profile.recommended_actions}

Current Page Context for website: {website}
- Accessibility Snapshot (first 2000 chars): {baseline_snapshot[:2000] if baseline_snapshot else 'Not available'}

Generate CSS and JavaScript adaptations for this homepage that will:
1. Address the specific needs described in the UX profile
2. Create measurable visual changes for scientific comparison
3. Improve accessibility for this user profile
4. Follow best practices for the profile category

Include clear purposes for each adaptation.
Ensure CSS uses !important where necessary.
""".strip()

        messages = [
            {
                "role": "system",
                "content": """You are a creative UX accessibility expert specializing in innovative web adaptations for diverse user needs.

Generate comprehensive CSS and JavaScript code to adapt the current homepage for the specific user profile.

BE CREATIVE AND SPECIFIC! Examples of innovative adaptations:

For Low Vision:
- Dramatically increase font sizes (e.g., body { font-size: 200% !important; })
- Add high-contrast themes (e.g., body { background-color: black !important; color: yellow !important; })
- Create magnification effects on hover for key elements
- Add prominent, thick focus indicators (e.g., *:focus { outline: 3px solid blue !important; })

For Motion Sensitivity:
- Remove all animations and transitions (e.g., * { transition: none !important; animation: none !important; })
- Add prefers-reduced-motion CSS
- Disable auto-playing content (JS may be needed)

For Reading Disorders:
- Change fonts to dyslexia-friendly ones (e.g., body { font-family: 'OpenDyslexic', sans-serif !important; })
- Increase line spacing and letter spacing (e.g., p { line-height: 1.8 !important; letter-spacing: 0.1em !important; })
- Add reading guides or highlighting via JS

For Photophobia:
- Apply dark filters or overlays (e.g., body::before { content:''; position:fixed; top:0;left:0;width:100%;height:100%; background:rgba(0,0,0,0.5); z-index:9999; pointer-events:none;})
- Reduce brightness significantly
- Add blue light filters (e.g., body { filter: sepia(0.2) contrast(0.9) brightness(0.8) hue-rotate(-10deg) !important; })

For ADHD/Attention Issues:
- Remove distracting animations (similar to motion sensitivity)
- Simplify layouts dramatically (e.g., hide non-essential sections with display:none !important;)
- Add focus management JS to guide users

Important guidelines:
- Make adaptations that are VISUALLY DRAMATIC and measurable.
- Use !important in CSS to ensure overrides work effectively.
- Target specific elements if possible using info from the accessibility snapshot (though full DOM is not provided, general tags can be used).
- Think scientifically - what changes can be measured for impact?

Respond ONLY in JSON format:
{
    "css": "/* CSS code with comments */",
    "css_purpose": "Clear explanation of what the CSS achieves for the profile",
    "js": "// JavaScript code with comments",
    "js_purpose": "Clear explanation of what the JS achieves for theprofile"
}"""
            }
        ]
        user_message_content = []
        if screenshot_data and _is_valid_base64_png(screenshot_data):
            user_message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}})
        user_message_content.append({"type": "text", "text": user_content_text})
        messages.append({"role": "user", "content": user_message_content})

        start_time = time.time()
        content = ""
        try:
            response, metrics = await self.openrouter.get_completion(
                model, messages,
                max_tokens=1800,
                analysis_type="ux_adaptation",
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "UXAdaptation",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "css": {"type": "string"},
                                "css_purpose": {"type": "string"},
                                "js": {"type": "string"},
                                "js_purpose": {"type": "string"}
                            },
                            "required": ["css", "css_purpose", "js", "js_purpose"]
                        }
                    }
                }
            )
            try:
                print(f"    ⏱️ ux_adaptation LLM time={metrics.response_time:.2f}s; tokens: prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens}")
            except Exception:
                pass

            content = (response.get('choices', [{}])[0].get('message', {}).get('content', '') or '')
            result_json = extract_first_json(content, context='ux_adaptation')
            if result_json is None:
                # Retry with stricter instruction
                retry_messages = [
                    {
                        "role": "system",
                        "content": """Return ONLY a valid JSON object. Use these exact keys: css, css_purpose, js, js_purpose.

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- Escape ALL quotes in CSS/JS code using \\\"
- Keep CSS under 800 characters, JS under 400 characters
- Use simple CSS rules only (color, font-size, background)
- Use minimal JS (simple DOM modifications)
- NO line breaks in CSS/JS strings - use spaces instead
- NO complex selectors or multi-line CSS
- TEST your JSON before responding

Example format:
{"css":"body{color:red !important;}","css_purpose":"Changes text color","js":"","js_purpose":"No JS needed"}"""
                    },
                    {"role": "user", "content": f"Generate accessibility CSS/JS for {ux_profile.name}. Valid JSON only."}
                ]
                response2, metrics2 = await self.openrouter.get_completion(
                    model, retry_messages, max_tokens=800,
                    analysis_type="ux_adaptation_retry", temperature=0.0
                )
                try:
                    print(f"    ⏱️ ux_adaptation retry LLM time={metrics2.response_time:.2f}s; tokens: prompt={metrics2.prompt_tokens}, completion={metrics2.completion_tokens}")
                except Exception:
                    pass
                content2 = (response2.get('choices', [{}])[0].get('message', {}).get('content', '') or '')
                result_json = extract_first_json(content2, context='ux_adaptation_retry')
                if result_json is None:
                    raw_path = self._save_raw_llm("raw_ux_adaptation_response", content2)
                    raise ValueError(f"Non-JSON response from model after retry. Saved preview to {raw_path or 'N/A'}")
                metrics = metrics2

            return (
                result_json.get('css', ''),
                result_json.get('js', ''),
                result_json.get('css_purpose', 'No CSS purpose provided'),
                result_json.get('js_purpose', 'No JS purpose provided'),
                metrics
            )
        except Exception as e:
            print(f"    Failed to generate UX adaptations: {e}")
            try:
                raw_path = self._save_raw_llm("raw_ux_adaptation_response", content if 'content' in locals() else "")
            except Exception:
                raw_path = None
            try:
                elapsed = time.time() - start_time
            except Exception:
                elapsed = 0.0
            error_metric = LLMMetrics(model=model, analysis_type="ux_adaptation", timestamp=datetime.now().isoformat(), response_time=elapsed)
            print(f"    ⏱️ ux_adaptation error LLM time={elapsed:.2f}s; returning fallback")
            return "", "", "Error generating CSS", "Error generating JS", error_metric

    async def apply_ux_profile(self, ux_profile: UXProfile, website: str, model: str = "openai/gpt-4o-mini", screenshot_data: Optional[str] = None, baseline_snapshot: str = "") -> Tuple[List[Dict], Optional[LLMMetrics]]:
        if not self.session:
            return [], None
        css, js, css_purpose, js_purpose, generation_metrics = await self.generate_ux_adaptations(
            ux_profile, website, model, screenshot_data, baseline_snapshot
        )
        applied_adaptations = []
        if css:
            try:
                print(f"    Applying CSS for {ux_profile.name}: {css_purpose[:100]}...")
                await self.session.call_tool("browser_inject_css", {"css": css})
                applied_adaptations.append({"type": "css", "description": css_purpose, "code": css})
            except Exception as e:
                print(f"    Failed to apply CSS: {e}")
        if js:
            try:
                print(f"    Applying JS for {ux_profile.name}: {js_purpose[:100]}...")
                await self.session.call_tool("browser_inject_js", {"javascript": js})
                applied_adaptations.append({"type": "js", "description": js_purpose, "code": js})
            except Exception as e:
                print(f"    Failed to apply JS: {e}")
        return applied_adaptations, generation_metrics

    async def generate_comprehensive_visual_comparison(self, baseline_screenshot_b64: str, adapted_screenshot_b64: str,
                                                       baseline_html: str, adapted_html: str, ux_profile_name: str,
                                                       model: str = "openai/gpt-4o") -> Tuple[Optional[VisualComparison], List[LLMMetrics]]:
        collected_llm_metrics: List[LLMMetrics] = []
        print("    🔬 Generating comprehensive expert visual analysis...")
        visual_analysis_messages = [
            {
                "role": "system",
                "content": """You are a team of accessibility, WCAG, UX, and visual design experts. Analyze the provided baseline and adapted interface screenshots and HTML snippets. The adaptation is for a specific UX profile.

Provide a comprehensive analysis covering:
1.  **Accessibility Improvements**: Specific changes benefiting users (e.g., contrast, focus, ARIA).
2.  **WCAG Compliance Notes**: Observations on WCAG 2.1 AA/AAA relevant to the changes. Estimate a WCAG compliance score (0-100) for the *adapted version* based on the provided snippets and visual changes.
3.  **UX Enhancements**: How the adaptation impacts usability, task flow, and cognitive load for the target profile.
4.  **Visual Design Critique**: Aesthetic changes, clarity, and visual hierarchy improvements/drawbacks.
5.  **Adaptation Score (0-2)**:
    * 0: No effective adaptation or negative impact.
    * 1: Partial or minor positive adaptation for the profile.
    * 2: Significant and effective adaptation for the profile.
    Provide clear reasoning for this score.
6.  **Visual Indicators**: Suggest 2-3 key areas on the *adapted screenshot* that demonstrate significant changes using descriptive text (e.g., "Increased font size of main navigation", "Simplified header section").
7.  **Detailed Notes**: Summarize overall findings.

Respond ONLY in JSON format:
{
    "accessibility_expert": {
        "wcag_compliance_score": 75.0, // Estimated for adapted version
        "accessibility_improvements": ["Improved color contrast", "Better focus indicators"],
        "accessibility_issues_remaining": ["Some images may still lack alt text (cannot verify from snippet)"],
        "recommendations": ["Review semantic HTML for all interactive elements."],
        "detailed_reasoning": "The adapted version shows better contrast..."
    },
    "wcag_expert": { // Focus on observed changes
        "level_a_compliance_observed": 85.0,
        "level_aa_compliance_observed": 70.0,
        "level_aaa_compliance_observed": 40.0, // If applicable
        "specific_criteria_analysis": {
            "1.4.3_contrast_minimum": "Adapted version appears to meet AA for text in screenshot.",
            "2.4.7_focus_visible": "Focus indicators seem more prominent if changed."
        },
        "priority_fixes_suggested": ["Verify all form inputs have associated labels."]
    },
    "ux_expert": {
        "ux_score_change_estimate": 8.0, // Estimated improvement e.g. +8 points on a 100-point scale
        "user_journey_impact": "Navigation likely easier for users with low vision due to larger text.",
        "cognitive_load_analysis": "Reduced clutter should lower cognitive load for users with attention difficulties.",
        "recommendations": ["Test with actual users matching the profile."]
    },
    "visual_critic": {
        "visual_design_score_change_estimate": 7.0, // Estimated improvement
        "typography_analysis": "Font changes improve readability.",
        "color_scheme_analysis": "New color scheme is more accessible but check brand alignment.",
        "design_critique": "Overall, the adapted design is cleaner, though some visual appeal might be lost depending on the adaptation."
    },
    "adaptation_score": 1, // 0, 1, or 2
    "adaptation_reasoning": "The adaptation partially addresses the needs of the profile by increasing font size, but lacks advanced features like text-to-speech integration.",
    "specific_improvements_noted": ["Increased font size", "Higher contrast mode applied"],
    "missed_opportunities_noted": ["No changes to keyboard navigation observed", "ARIA attributes not added (as far as visible)"],
    "accessibility_impact_summary": "Moderate positive impact on accessibility for the target profile.",
    "ux_impact_summary": "Noticeable improvement in usability for the target profile.",
    "profile_alignment_summary": "Partially aligned with the UX profile.",
    "implementation_quality_summary": "CSS changes seem robust with !important, JS changes (if any) not assessable without code.",
    "visual_indicators_on_adapted": [ // Describe areas on ADAPTED screenshot
        {"type": "highlight_area", "description": "Main content text significantly larger and re-spaced.", "coordinates_placeholder": "[x,y,w,h] (conceptual)"},
        {"type": "focus_change", "description": "Navigation links show clearer focus style.", "coordinates_placeholder": "[x,y,w,h] (conceptual)"}
    ],
    "visual_differences_summary": [ // Key differences observed
        {"category": "typography", "description": "Font size increased globally.", "impact": "Improved readability for low vision."},
        {"category": "color", "description": "High contrast mode applied (black background, yellow text).", "impact": "Significantly better for users needing high contrast."}
    ],
    "improvement_score_overall": 7.5, // Subjective overall improvement (0-10)
    "scientific_observations_summary": "The adaptation demonstrates measurable changes in text size and contrast ratios, which are expected to positively correlate with improved task completion times for users with specific visual impairments.",
    "detailed_notes_summary": "This adaptation is a good step..."
}"""
            }
        ]
        user_content_list = [
            {
                "type": "text",
                "text": f"Compare these interfaces. The adaptation is for the '{ux_profile_name}' UX profile.\n\nBaseline HTML (first 2000 chars):\n{baseline_html[:2000]}\n\nAdapted HTML (first 2000 chars):\n{adapted_html[:2000]}"
            }
        ]
        if baseline_screenshot_b64 and _is_valid_base64_png(baseline_screenshot_b64):
            user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{baseline_screenshot_b64}"}})
        if adapted_screenshot_b64 and _is_valid_base64_png(adapted_screenshot_b64):
            user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{adapted_screenshot_b64}"}})
        visual_analysis_messages.append({"role": "user", "content": user_content_list})

        llm_analysis_json = {}
        max_retries = 3
        models_to_try = [model]
        for current_model in models_to_try:
            print(f"    🔄 Trying visual analysis with model: {current_model}")
            for attempt in range(max_retries):
                try:
                    response, metrics = await self.openrouter.get_completion(
                        current_model, visual_analysis_messages,
                        max_tokens=2000, analysis_type="visual_comparison",
                        temperature=0.2, response_format={"type": "json_object"}, max_retries=2
                    )
                    collected_llm_metrics.append(metrics)
                    content = (response.get('choices', [{}])[0].get('message', {}).get('content', '') or '')
                    if not content.strip():
                        raise ValueError("Empty content from LLM")
                    parsed = extract_first_json(content, context='visual_comparison')
                    if parsed is None:
                        raw_path = self._save_raw_llm("raw_visual_comparison_response", content)
                        raise ValueError(f"Non-JSON response from model. Saved preview to {raw_path or 'N/A'}")
                    llm_analysis_json = parsed
                    print(f"    ✅ Visual analysis successful with model: {current_model}")
                    break
                except Exception as e:
                    logger.warning(f"Visual analysis attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.5 ** attempt)
                    else:
                        print(f"    ⚠️ Model {current_model} failed after {max_retries} attempts")
            if llm_analysis_json:
                break
        else:
            logger.error(f"Failed to get LLM visual analysis with all models: {models_to_try}")
            try:
                raw_path = self._save_raw_llm("raw_visual_comparison_response", content if 'content' in locals() else "")
            except Exception:
                raw_path = None
            error_metric = LLMMetrics(model=model, analysis_type="visual_comparison_error", timestamp=datetime.now().isoformat())
            collected_llm_metrics.append(error_metric)
            llm_analysis_json = {
                "error": "All visual analysis models failed",
                "raw_saved_path": raw_path or "",
                "adaptation_score": 1,
                "adaptation_reasoning": "Fallback estimate (models failed)",
                "accessibility_expert": {"wcag_compliance_score": 60.0},
                "visual_differences_summary": [{"category": "adaptation", "description": "Adaptations applied", "impact": "Expected positive"}],
                "improvement_score_overall": 5.0,
                "scientific_observations_summary": "Fallback summary due to LLM failure",
                "detailed_notes_summary": "Fallback notes"
            }

        css_js_analysis_json = {}
        css_js_metrics = None
        if adapted_html:
            css_js_analysis_json, css_js_metrics = await self.analyze_css_js_with_llm(adapted_html, ux_profile_name, model)
            if css_js_metrics:
                collected_llm_metrics.append(css_js_metrics)

        expert_analysis_data = ExpertAnalysis(
            accessibility_expert=llm_analysis_json.get('accessibility_expert', {}),
            wcag_expert=llm_analysis_json.get('wcag_expert', {}),
            ux_expert=llm_analysis_json.get('ux_expert', {}),
            visual_critic=llm_analysis_json.get('visual_critic', {}),
            css_analysis=css_js_analysis_json.get('css_analysis', {}),
            js_analysis=css_js_analysis_json.get('js_analysis', {})
        )
        adaptation_score_val = llm_analysis_json.get('adaptation_score', 0)
        adaptation_score_data = AdaptationScore(
            score=int(adaptation_score_val),
            reasoning=str(llm_analysis_json.get('adaptation_reasoning', 'N/A')),
            specific_improvements=list(llm_analysis_json.get('specific_improvements_noted', [])),
            missed_opportunities=list(llm_analysis_json.get('missed_opportunities_noted', [])),
            wcag_compliance_score=float(llm_analysis_json.get('accessibility_expert', {}).get('wcag_compliance_score', 0.0)),
            accessibility_impact=str(llm_analysis_json.get('accessibility_impact_summary', 'N/A')),
            user_experience_impact=str(llm_analysis_json.get('ux_impact_summary', 'N/A')),
            profile_alignment=str(llm_analysis_json.get('profile_alignment_summary', 'N/A')),
            implementation_quality=str(llm_analysis_json.get('implementation_quality_summary', 'N/A'))
        )

        visual_comparison_obj = VisualComparison(
            baseline_screenshot_path="baseline_screenshot.png",
            adapted_screenshot_path=f"adapted_screenshot_{ux_profile_name.replace(' ', '_') if ux_profile_name else 'default'}.png",
            baseline_screenshot_b64=baseline_screenshot_b64,
            adapted_screenshot_b64=adapted_screenshot_b64,
            comparison_analysis=f"AI-powered analysis of interface adaptations for {ux_profile_name or 'baseline'}",
            visual_differences=list(llm_analysis_json.get('visual_differences_summary', [])),
            improvement_score=float(llm_analysis_json.get('improvement_score_overall', 0.0)),
            scientific_observations=str(llm_analysis_json.get('scientific_observations_summary', 'N/A')),
            expert_analysis=expert_analysis_data,
            adaptation_score=adaptation_score_data,
            visual_indicators=list(llm_analysis_json.get('visual_indicators_on_adapted', [])),
            detailed_notes=str(llm_analysis_json.get('detailed_notes_summary', 'N/A'))
        )
        return visual_comparison_obj, collected_llm_metrics

    async def save_test_artifacts(self, test_identifier: str, baseline_screenshot_b64: str, baseline_snapshot: str,
                                  adapted_screenshot_b64: str = "", adapted_snapshot: str = "",
                                  ux_profile_name: str = "", bot_detected: bool = False,
                                  visual_comparison: Optional[VisualComparison] = None,
                                  ux_adaptations: Optional[List[Dict]] = None,
                                  baseline_metrics: Optional[HomepageMetrics] = None,
                                  adapted_metrics: Optional[HomepageMetrics] = None,
                                  current_test_llm_metrics: Optional[List[LLMMetrics]] = None,
                                  models_used: Optional[Dict[str, str]] = None,
                                  website: str = ""):
        site = website or ""
        try:
            site = site.split("://", 1)[-1]
        except Exception:
            pass
        site = site.strip().strip('/')
        safe_site = re.sub(r'[^a-zA-Z0-9._-]+', '-', site) if site else "site"
        dir_name = f"{safe_site}_{str(test_identifier)}"
        test_dir = self.output_dir / dir_name
        test_dir.mkdir(parents=True, exist_ok=True)

        baseline_screenshot_size = 0
        baseline_snapshot_size = 0
        adapted_screenshot_size = 0
        adapted_snapshot_size = 0

        baseline_screenshot_filename = "baseline_screenshot.png"
        baseline_snapshot_filename = "baseline_snapshot.html"
        adapted_screenshot_filename = f"adapted_screenshot_{ux_profile_name.replace(' ', '_') if ux_profile_name else 'default'}.png"
        adapted_snapshot_filename = f"adapted_snapshot_{ux_profile_name.replace(' ', '_') if ux_profile_name else 'default'}.html"

        if baseline_screenshot_b64:
            baseline_screenshot_size = self.save_screenshot(baseline_screenshot_b64, test_dir / baseline_screenshot_filename)
        if baseline_snapshot:
            baseline_snapshot_size = self.save_html_snapshot(baseline_snapshot, test_dir / baseline_snapshot_filename)

        if adapted_screenshot_b64:
            adapted_screenshot_size = self.save_screenshot(adapted_screenshot_b64, test_dir / adapted_screenshot_filename)
        if adapted_snapshot:
            adapted_snapshot_size = self.save_html_snapshot(adapted_snapshot, test_dir / adapted_snapshot_filename)

        analysis_data = {
            "test_number": test_identifier,
            "timestamp": datetime.now().isoformat(),
            "ux_profile": ux_profile_name,
            "bot_detection_encountered": bot_detected,
            "visual_comparison": asdict(visual_comparison) if visual_comparison else None,
            "ux_adaptations": ux_adaptations if ux_adaptations else [],
            "baseline_metrics": asdict(baseline_metrics) if baseline_metrics else {},
            "adapted_metrics": asdict(adapted_metrics) if adapted_metrics else {},
            "llm_metrics_for_test": [asdict(metric) for metric in (current_test_llm_metrics or [])],
            "models_used": (models_used or {})
        }
        with open(test_dir / "comprehensive_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        info_path = test_dir / "test_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f" WebEval Test {test_identifier}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"UX Profile: {ux_profile_name or 'N/A'}\n")
            f.write(f"Bot Detection Encountered: {bot_detected}\n")
            f.write(f"Baseline Screenshot: {baseline_screenshot_filename} ({baseline_screenshot_size} bytes)\n")
            f.write(f"Baseline Snapshot: {baseline_snapshot_filename} ({baseline_snapshot_size} bytes)\n")
            if adapted_screenshot_size > 0:
                f.write(f"Adapted Screenshot: {adapted_screenshot_filename} ({adapted_screenshot_size} bytes)\n")
            if adapted_snapshot_size > 0:
                f.write(f"Adapted Snapshot: {adapted_snapshot_filename} ({adapted_snapshot_size} bytes)\n")
            if visual_comparison and visual_comparison.adaptation_score:
                f.write(f"Adaptation Score: {visual_comparison.adaptation_score.score}/2\n")
                f.write(f"WCAG Compliance (Adapted): {visual_comparison.adaptation_score.wcag_compliance_score}%\n")
            if baseline_metrics:
                f.write(f"Baseline Accessibility Score: {baseline_metrics.accessibility_score:.1f}\n")
            if adapted_metrics:
                f.write(f"Adapted Accessibility Score: {adapted_metrics.accessibility_score:.1f}\n")
            if current_test_llm_metrics:
                total_tokens_test = sum(m.total_tokens for m in current_test_llm_metrics)
                f.write(f"LLM Tokens for this test: {total_tokens_test:,}\n")

    async def execute_test(self, test_case: TestCase) -> HomepageResult:
        self.current_test_num += 1
        print(f"\n🚀 Executing  WebEval Test {self.current_test_num}: {test_case.website}")
        print(f"  🎯 UX Profile: {test_case.ux_profile or 'None'}")

        gen_model = getattr(test_case, 'gen_llm_model', None) or getattr(test_case, 'generation_model', None) or test_case.llm_model
        eval_model = getattr(test_case, 'eval_llm_model', None) or getattr(test_case, 'evaluation_model', None) or test_case.llm_model
        print(f"  🤖 LLM Models → generation: {gen_model} | evaluation: {eval_model}")

        test_start_time = datetime.now()
        current_test_llm_metrics: List[LLMMetrics] = []

        status = "pending"
        execution_time_secs = 0.0
        baseline_screenshot_b64 = ""
        adapted_screenshot_b64 = ""
        baseline_snapshot_html = ""
        adapted_snapshot_html = ""
        final_baseline_metrics = HomepageMetrics()
        final_adapted_metrics: Optional[HomepageMetrics] = None
        final_ux_adaptations: List[Dict] = []
        bot_detection_was_encountered = False
        final_bot_detection_result: Optional[BotDetectionResult] = None
        final_visual_comparison: Optional[VisualComparison] = None
        error_msg: Optional[str] = None
        load_time_secs: float = 0.0

        try:
            try:
                self.update_csv_status(test_case.website, "running")
            except Exception:
                pass
            # Connection is now managed by the CLI to avoid async context issues
            # Only connect here if no session exists (fallback)
            if not self.session:
                try:
                    await self.connect_to_playwright_mcp()
                except Exception as e:
                    print(f"  ⚠️ Could not establish Playwright MCP connection: {e}")
                    # Continue anyway - some tests might work without browser

            print(f"  🌐 Navigating to {test_case.website}...")
            load_time_secs, bot_detection_was_encountered, final_bot_detection_result = await self.navigate_to_url(test_case.website, current_test_llm_metrics, eval_model)

            if bot_detection_was_encountered and not (final_bot_detection_result and not final_bot_detection_result.is_bot_detected):
                print(f"  ⚠️ Bot detection blocked access or could not be bypassed: {test_case.website}")
                status = "blocked"
                error_msg = f"Bot detection: {final_bot_detection_result.detection_type if final_bot_detection_result else 'Unknown'}"
                baseline_screenshot_b64 = await self.take_screenshot()
                baseline_snapshot_html = await self.get_page_snapshot()
                metrics_obj, metrics_llm = await self.analyze_homepage_metrics(baseline_snapshot_html, baseline_screenshot_b64, eval_model)
                if metrics_llm: current_test_llm_metrics.append(metrics_llm)
                final_baseline_metrics = metrics_obj
                final_baseline_metrics.load_time = load_time_secs
                if baseline_screenshot_b64:
                    final_baseline_metrics.screenshot_size_bytes = _safe_b64_len(baseline_screenshot_b64)
                if baseline_snapshot_html:
                    final_baseline_metrics.html_size_bytes = len(baseline_snapshot_html.encode('utf-8'))
            else:
                print("  📸 Capturing baseline state...")
                baseline_screenshot_b64 = await self.take_screenshot()
                baseline_snapshot_html = await self.get_page_snapshot()

                print("  🔍 Analyzing baseline metrics...")
                metrics_obj, metrics_llm = await self.analyze_homepage_metrics(baseline_snapshot_html, baseline_screenshot_b64, eval_model)
                if metrics_llm: current_test_llm_metrics.append(metrics_llm)
                final_baseline_metrics = metrics_obj
                final_baseline_metrics.load_time = load_time_secs
                if baseline_screenshot_b64:
                    final_baseline_metrics.screenshot_size_bytes = _safe_b64_len(baseline_screenshot_b64)
                if baseline_snapshot_html:
                    final_baseline_metrics.html_size_bytes = len(baseline_snapshot_html.encode('utf-8'))

                if test_case.ux_profile and test_case.ux_profile in self.ux_profiles:
                    active_ux_profile = self.ux_profiles[test_case.ux_profile]
                    print(f"  🎨 Applying UX profile: {active_ux_profile.name}")

                    applied_adaptations, adaptation_llm_metrics = await self.apply_ux_profile(
                        active_ux_profile, test_case.website, gen_model, baseline_screenshot_b64, baseline_snapshot_html
                    )
                    if adaptation_llm_metrics: current_test_llm_metrics.append(adaptation_llm_metrics)
                    final_ux_adaptations.extend(applied_adaptations)

                    if applied_adaptations:
                        await self.session.call_tool("browser_wait_for", {"time": 3})

                        print("  📸 Capturing adapted state...")
                        adapted_screenshot_b64 = await self.take_screenshot()
                        adapted_snapshot_html = await self.get_page_snapshot()

                        print("  🔍 Analyzing adapted metrics...")
                        adapted_metrics_obj, adapted_metrics_llm = await self.analyze_homepage_metrics(adapted_snapshot_html, adapted_screenshot_b64, eval_model)
                        if adapted_metrics_llm: current_test_llm_metrics.append(adapted_metrics_llm)
                        final_adapted_metrics = adapted_metrics_obj
                        if adapted_screenshot_b64:
                            final_adapted_metrics.screenshot_size_bytes = _safe_b64_len(adapted_screenshot_b64)
                        if adapted_snapshot_html:
                            final_adapted_metrics.html_size_bytes = len(adapted_snapshot_html.encode('utf-8'))

                        if baseline_screenshot_b64 and adapted_screenshot_b64:
                            print("  🔬 Generating comprehensive visual comparison analysis...")
                            vis_comp_obj, vis_comp_llms = await self.generate_comprehensive_visual_comparison(
                                baseline_screenshot_b64, adapted_screenshot_b64,
                                baseline_snapshot_html, adapted_snapshot_html,
                                active_ux_profile.name, eval_model
                            )
                            if vis_comp_llms: current_test_llm_metrics.extend(vis_comp_llms)
                            final_visual_comparison = vis_comp_obj
                status = "success"

        except ConnectionError as ce:
            print(f"  ❌ Test {self.current_test_num} failed due to connection error: {ce}")
            error_msg = f"ConnectionError: {ce}"
            status = "error"
        except Exception as e:
            print(f"  ❌ Test {self.current_test_num} failed with unexpected error: {e}")
            import traceback; traceback.print_exc()
            error_msg = f"Unexpected error: {type(e).__name__} - {e}"
            status = "error"
            try:
                if not baseline_screenshot_b64: baseline_screenshot_b64 = await self.take_screenshot()
                if not baseline_snapshot_html: baseline_snapshot_html = await self.get_page_snapshot()
            except Exception as capture_e:
                print(f"    Could not capture final state during error handling: {capture_e}")

        execution_time_secs = (datetime.now() - test_start_time).total_seconds()

        await self.save_test_artifacts(
            getattr(test_case, "case_id", f"TC_{self.current_test_num:04d}"),
            baseline_screenshot_b64, baseline_snapshot_html,
            adapted_screenshot_b64, adapted_snapshot_html, test_case.ux_profile,
            bot_detection_was_encountered, final_visual_comparison, final_ux_adaptations,
            final_baseline_metrics, final_adapted_metrics, current_test_llm_metrics,
            models_used={"generation": gen_model, "evaluation_metrics": eval_model, "evaluation_visual": eval_model},
            website=test_case.website
        )

        result = HomepageResult(
            test_case=test_case,
            status=status,
            execution_time=execution_time_secs,
            baseline_screenshot=baseline_screenshot_b64,
            adapted_screenshot=adapted_screenshot_b64,
            baseline_snapshot=baseline_snapshot_html,
            adapted_snapshot=adapted_snapshot_html,
            baseline_metrics=final_baseline_metrics,
            adapted_metrics=final_adapted_metrics,
            ux_adaptations=final_ux_adaptations,
            bot_detection_encountered=bot_detection_was_encountered,
            bot_detection_result=final_bot_detection_result,
            visual_comparison=final_visual_comparison,
            error_message=error_msg,
            llm_metrics=current_test_llm_metrics,
            test_timestamp=datetime.now().isoformat(),
            detailed_analysis={
                "load_time_seconds": load_time_secs,
                "total_llm_calls_this_test": len(current_test_llm_metrics),
                "total_tokens_this_test": sum(m.total_tokens for m in current_test_llm_metrics),
                "profile_applied": bool(test_case.ux_profile and test_case.ux_profile in self.ux_profiles),
                "adaptations_actually_applied_count": len(final_ux_adaptations)
            }
        )

        self.results.append(result)
        self.all_llm_metrics.extend(current_test_llm_metrics)

        if status == "success":
            print(f"  ✅ Test {self.current_test_num} completed successfully in {execution_time_secs:.2f}s")
            if final_visual_comparison and final_visual_comparison.adaptation_score:
                print(f"    ⭐ Adaptation Score: {final_visual_comparison.adaptation_score.score}/2")
                try:
                    print(f"    📈 WCAG Compliance (adapted): {final_visual_comparison.adaptation_score.wcag_compliance_score:.1f}%")
                except Exception:
                    pass
        else:
            print(f"  ⚠️ Test {self.current_test_num} finished with status={status} in {execution_time_secs:.2f}s")
            if error_msg:
                print(f"    Error: {error_msg}")

        try:
            self.update_csv_status(test_case.website, "completed" if status == "success" else status)
        except Exception:
            pass

        return result

    async def run_tests(self, test_cases: List[TestCase]) -> List[HomepageResult]:
        """
        Run a sequence of tests, handling MCP lifecycle and per-test delays.
        Returns the accumulated self.results list.
        """
        if not test_cases:
            print("No test cases provided to run_tests().")
            return []

        # Note: Connection and disconnection are now handled by the caller (CLI)
        # to avoid asyncio context issues
        
        for idx, tc in enumerate(test_cases, 1):
            try:
                await asyncio.sleep(0.2)
                await self.execute_test(tc)
            except Exception as e:
                print(f"❌ Unhandled error during test {idx} ({tc.website}): {e}")
            finally:
                # Small pacing between tests to avoid rate/bot triggers
                await asyncio.sleep(0.5)

        return self.results

    def save_overall_results(self) -> None:
        """
        Persist aggregate outputs:
        - overall_results.json: full dataclass dump
        - overall_results.csv: summary table
        - overall_report.html: rich HTML report via report_utils
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # JSON aggregate
        try:
            out_json = self.output_dir / "overall_results.json"
            serializable = [asdict(r) for r in self.results]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)
            print(f"📦 Saved overall JSON: {out_json}")
        except Exception as e:
            print(f"❌ Failed to save overall JSON: {e}")

        # CSV aggregate
        try:
            out_csv = self.output_dir / "overall_results.csv"
            fieldnames = [
                "website", "status", "execution_time", "ux_profile",
                "baseline_accessibility", "adapted_accessibility",
                "bot_detection_type", "error_message",
                "total_llm_tokens"
            ]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    total_tokens = sum(m.total_tokens for m in (r.llm_metrics or []))
                    row = {
                        "website": r.test_case.website,
                        "status": r.status,
                        "execution_time": f"{r.execution_time:.2f}",
                        "ux_profile": r.test_case.ux_profile or "",
                        "baseline_accessibility": getattr(r.baseline_metrics, "accessibility_score", 0.0) if r.baseline_metrics else "",
                        "adapted_accessibility": getattr(r.adapted_metrics, "accessibility_score", "") if r.adapted_metrics else "",
                        "bot_detection_type": (r.bot_detection_result.detection_type if r.bot_detection_result else ""),
                        "error_message": r.error_message or "",
                        "total_llm_tokens": total_tokens,
                    }
                    writer.writerow(row)
            print(f"📄 Saved overall CSV: {out_csv}")
        except Exception as e:
            print(f"❌ Failed to save overall CSV: {e}")

        # HTML report
        try:
            out_html = self.output_dir / "overall_report.html"
            # Include statistical figures/tables if present
            stats_dir = self.output_dir / "statistical_analysis"
            html = generate_comprehensive_report_html(self.results, stats_dir if stats_dir.exists() else None)
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"🖼️ Saved overall HTML report: {out_html}")
        except Exception as e:
            print(f"❌ Failed to save overall HTML report: {e}")

        # Statistical CSVs for paper-ready aggregation
        try:
            # Global run stats
            total_tests = len(self.results)
            successful = sum(1 for r in self.results if r.status == "success")
            blocked = sum(1 for r in self.results if r.status == "blocked")
            failed = sum(1 for r in self.results if r.status == "error")
            adapted_tests = sum(1 for r in self.results if (r.ux_adaptations and len(r.ux_adaptations) > 0))
            total_tokens = sum(m.total_tokens for r in self.results for m in (r.llm_metrics or []))

            with open(self.output_dir / "overall_run_stats.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                w.writerow(["total_tests", total_tests])
                w.writerow(["successful_tests", successful])
                w.writerow(["blocked_tests", blocked])
                w.writerow(["failed_tests", failed])
                w.writerow(["adapted_tests", adapted_tests])
                w.writerow(["total_llm_tokens", total_tokens])

            # Per-model LLM aggregation
            per_model: Dict[str, Dict] = {}
            for m in self.all_llm_metrics:
                d = per_model.setdefault(m.model, {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "sum_resp_time": 0.0,
                })
                d["calls"] += 1
                d["prompt_tokens"] += m.prompt_tokens
                d["completion_tokens"] += m.completion_tokens
                d["total_tokens"] += m.total_tokens
                d["sum_resp_time"] += m.response_time

            with open(self.output_dir / "overall_model_stats.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["model", "calls", "prompt_tokens", "completion_tokens", "total_tokens", "avg_response_time"])
                for model, d in sorted(per_model.items(), key=lambda kv: kv[0]):
                    avg_rt = (d["sum_resp_time"] / d["calls"]) if d["calls"] else 0.0
                    w.writerow([model, d["calls"], d["prompt_tokens"], d["completion_tokens"], d["total_tokens"], f"{avg_rt:.3f}"])

            # Per-profile accessibility delta (adapted - baseline)
            profile_delta: Dict[str, Dict[str, float]] = {}
            for r in self.results:
                profile = r.test_case.ux_profile or ""
                if not profile:
                    continue
                base = getattr(r.baseline_metrics, "accessibility_score", None) if r.baseline_metrics else None
                adap = getattr(r.adapted_metrics, "accessibility_score", None) if r.adapted_metrics else None
                if base is None or adap is None:
                    continue
                d = profile_delta.setdefault(profile, {"count": 0, "sum_delta": 0.0})
                d["count"] += 1
                d["sum_delta"] += (adap - base)

            with open(self.output_dir / "overall_profile_stats.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ux_profile", "n_with_adapted_metrics", "mean_accessibility_delta"])
                for prof, d in sorted(profile_delta.items(), key=lambda kv: kv[0]):
                    mean_delta = (d["sum_delta"] / d["count"]) if d["count"] else 0.0
                    w.writerow([prof, d["count"], f"{mean_delta:.3f}"])

            print(f"📊 Saved statistical CSVs in {self.output_dir}")
        except Exception as e:
            print(f"❌ Failed to save statistical CSVs: {e}")