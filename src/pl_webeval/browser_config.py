"""
Enhanced browser configuration for PL_WebEval with maximum compatibility and anti-bot detection.
Handles various browser options, stealth techniques, and error detection strategies.
"""

import random
from typing import Dict, List, Optional

class BrowserConfig:
    """Advanced configuration for browser compatibility and bot evasion."""
    
    # User agent strings for maximum compatibility - focusing on most common browsers
    USER_AGENTS = [
        # Chrome on Windows - Most Common (90%+ web traffic)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        
        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        
        # Edge (often bypasses some bot detection)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        
        # Safari (for sites that block Chrome)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        
        # Mobile Chrome (sometimes bypasses desktop bot detection)
        "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
        
        # Firefox as fallback
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    ]
    
    # Error page indicators that suggest unreachable website (not bot detection)
    ERROR_INDICATORS = [
        # Chrome/Chromium network errors
        "ERR_NAME_NOT_RESOLVED",
        "ERR_CONNECTION_REFUSED",
        "ERR_CONNECTION_TIMED_OUT",
        "ERR_ADDRESS_UNREACHABLE",
        "ERR_CONNECTION_RESET",
        "ERR_INTERNET_DISCONNECTED",
        "ERR_NETWORK_ACCESS_DENIED",
        "ERR_NAME_RESOLUTION_FAILED",
        "ERR_FAILED",
        "DNS_PROBE_FINISHED",
        "This site can't be reached",
        "took too long to respond",
        "The connection was reset",
        "ERR_CERT_AUTHORITY_INVALID",  
        
        # Firefox network errors
        "Server not found",
        "Unable to connect",
        "The connection has timed out",
        
        # Server errors (might be temporary)
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
    ]
    
    # Bot detection indicators (NOT unreachable - needs handling)
    BOT_DETECTION_INDICATORS = [
        # CAPTCHA systems
        "captcha",
        "CAPTCHA",
        "reCAPTCHA",
        "hCaptcha",
        "FunCaptcha",
        "GeeTest",
        
        # Cloudflare protection
        "Cloudflare",
        "cf-browser-verification",
        "Checking your browser",
        "Please wait while we check your browser",
        "DDoS protection by Cloudflare",
        "Ray ID:",
        "cf_clearance",
        
        # Generic bot detection
        "Access denied",
        "Access Denied",
        "ACCESS DENIED",
        "403 Forbidden",
        "Bot detection",
        "Security check",
        "One more step",
        "Verify you are human",
        "Please verify you are human",
        "Prove you are human",
        "I'm not a robot",
        "Please complete the security check",
        "Enable JavaScript",
        "Enable cookies",
        "Browser verification",
        "Suspicious activity",
        "Automated access",
        "Rate limited",
        "Too many requests",
        
        # Specific protection services
        "PerimeterX",
        "DataDome",
        "Imperva",
        "Akamai",
        "Sucuri",
        "AWS WAF",
        "Barracuda",
        
        # JavaScript challenges
        "Please enable JavaScript",
        "JavaScript is disabled",
        "This website requires JavaScript",
    ]
    
    @staticmethod
    def get_browser_args(headless: bool = True, browser_type: str = "chrome") -> List[str]:
        """
        Get optimized browser arguments for maximum compatibility and stealth.
        
        Args:
            headless: Whether to run browser in headless mode
            browser_type: Type of browser (chrome, firefox, edge)
            
        Returns:
            List of browser arguments
        """
        base_args = []
        
        if browser_type in ["chrome", "edge"]:
            # Core stealth arguments
            base_args = [
                # Disable automation indicators
                "--disable-blink-features=AutomationControlled",
                "--disable-features=AutomationControlled",
                
                # Core compatibility
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu-sandbox",
                "--disable-setuid-sandbox",
                
                # Window and display
                "--window-size=1920,1080",
                "--start-maximized",
                "--force-device-scale-factor=1",
                
                # Security and isolation (reduce but not eliminate)
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--allow-running-insecure-content",
                "--ignore-certificate-errors",
                "--allow-insecure-localhost",
                
                # Performance and behavior
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI",
                "--disable-ipc-flooding-protection",
                "--disable-hang-monitor",
                "--disable-prompt-on-repost",
                "--disable-domain-reliability",
                "--disable-component-update",
                "--disable-features=AudioServiceOutOfProcess",
                "--disable-features=CalculateNativeWinOcclusion",
                
                # Storage and cookies
                "--password-store=basic",
                "--use-mock-keychain",
                "--disable-features=SameSiteByDefaultCookies",
                "--disable-features=CookiesWithoutSameSiteMustBeSecure",
                "--disable-features=CrossSiteDocumentBlockingIfIsolating",
                "--disable-features=CrossSiteDocumentBlockingAlways",
                
                # Media and features
                "--disable-features=MediaRouter",
                "--disable-features=DialMediaRouteProvider",
                "--disable-features=AcceptCHFrame",
                "--disable-features=AutoExpandDetailsElement",
                "--disable-features=CertificateTransparencyComponentUpdater",
                "--disable-features=AvoidUnnecessaryBeforeUnloadCheckSync",
                "--disable-features=Translate",
                
                # Network and caching
                "--aggressive-cache-discard",
                "--disable-background-networking",
                "--disable-breakpad",
                "--disable-client-side-phishing-detection",
                "--disable-default-apps",
                "--disable-extensions",
                "--disable-features=ImprovedCookieControls",
                "--disable-features=LazyFrameLoading",
                "--disable-features=GlobalMediaControls",
                "--disable-features=DestroyProfileOnBrowserClose",
                "--disable-sync",
                "--metrics-recording-only",
                "--no-first-run",
                "--safebrowsing-disable-auto-update",
                
                # Permissions
                "--disable-permissions-api",
                "--disable-notifications",
                "--disable-push-api-background-mode",
                
                # Additional stealth
                "--disable-logging",
                "--disable-login-animations",
                "--disable-infobars",
                "--hide-scrollbars",
                "--mute-audio",
                "--no-default-browser-check",
                "--no-pings",
                
                # Language and locale
                "--lang=en-US",
                "--accept-lang=en-US,en;q=0.9",
            ]
            
            if headless:
                # Use new headless mode which is less detectable
                base_args.extend([
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-tools",
                ])
            else:
                # Non-headless specific
                base_args.extend([
                    "--disable-popup-blocking",
                    "--disable-automation",
                ])
                
        elif browser_type == "firefox":
            base_args = [
                "-width", "1920",
                "-height", "1080", 
                "-pref", "dom.webdriver.enabled=false",
                "-pref", "useAutomationExtension=false",
                "-pref", "general.useragent.override=" + BrowserConfig.get_random_user_agent(),
            ]
            if headless:
                base_args.append("-headless")
        
        return base_args
    
    @staticmethod
    def get_random_user_agent() -> str:
        """Get a random user agent string for bot evasion."""
        return random.choice(BrowserConfig.USER_AGENTS)
    
    @staticmethod
    def get_stealth_headers() -> Dict[str, str]:
        """Get stealth HTTP headers to appear more human."""
        return BrowserConfig.get_locale_headers("en-US,en;q=0.9")
    
    @staticmethod
    def is_error_page(html_content: str, screenshot_text: str = "") -> bool:
        """
        Check if the page content indicates an error or unreachable website.
        This is different from bot detection - these are actual network/DNS errors.
        
        Args:
            html_content: HTML content of the page
            screenshot_text: Any text extracted from screenshot
            
        Returns:
            True if error page detected, False otherwise
        """
        combined_content = (html_content + " " + screenshot_text).lower()
        
        # Check for network error indicators only (not bot detection)
        for indicator in BrowserConfig.ERROR_INDICATORS:
            if indicator.lower() in combined_content:
                # Double check it's not bot detection masquerading as error
                is_bot = False
                for bot_indicator in BrowserConfig.BOT_DETECTION_INDICATORS:
                    if bot_indicator.lower() in combined_content:
                        is_bot = True
                        break
                if not is_bot:
                    return True
        
        # Check for empty or minimal HTML that suggests failure
        if len(html_content.strip()) < 100:
            # But not if it's a redirect or loading page
            if "redirect" not in combined_content and "loading" not in combined_content:
                return True
            
        # Check for browser's default error page structure
        if "chrome-error://" in combined_content or "about:neterror" in combined_content:
            return True
            
        return False
    
    @staticmethod
    def is_bot_detection_page(html_content: str, screenshot_text: str = "") -> bool:
        """
        Check if the page shows bot detection (not network error).
        
        Args:
            html_content: HTML content of the page
            screenshot_text: Any text extracted from screenshot
            
        Returns:
            True if bot detection detected, False otherwise
        """
        combined_content = (html_content + " " + screenshot_text).lower()
        
        for indicator in BrowserConfig.BOT_DETECTION_INDICATORS:
            if indicator.lower() in combined_content:
                return True
                
        return False
    
    @staticmethod
    def get_locale_headers(accept_language: str) -> Dict[str, str]:
        """Build stealth headers for a given Accept-Language chain."""
        return {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": accept_language,
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="131", "Google Chrome";v="131"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

    @staticmethod
    def get_mobile_user_agent(profile: str = "android_chrome") -> str:
        """Return a realistic mobile user agent."""
        if profile == "android_chrome":
            return "Mozilla/5.0 (Linux; Android 13; SM-G996N Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/131.0.0.0 Mobile Safari/537.36"
        if profile == "ios_safari":
            return "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
        return BrowserConfig.USER_AGENTS[0]

    @staticmethod
    def get_domain_overrides(url: str) -> Dict[str, object]:
        """
        Domain-specific overrides to improve reachability and reduce bot detection.
        Returns keys: user_agent(str), headers(dict), viewport(dict), locale(str), timezone(str), js_snippets(list[str]), ch_headers(dict)
        """
        u = (url or "").lower()
        overrides: Dict[str, object] = {}

        # Coupang uses aggressive bot mitigation (Akamai/PerimeterX variants). Use KR mobile fingerprint.
        if "coupang.com" in u:
            ua = BrowserConfig.get_mobile_user_agent("android_chrome")
            headers = BrowserConfig.get_locale_headers("ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7")
            # Override client hints to mobile Android
            headers.update({
                "sec-ch-ua-mobile": "?1",
                "sec-ch-ua-platform": '"Android"',
            })
            overrides = {
                "user_agent": ua,
                "headers": headers,
                "viewport": {"width": 390, "height": 844, "deviceScaleFactor": 3},
                "locale": "ko-KR",
                "timezone": "Asia/Seoul",
                "js_snippets": [
                    # Spoof language/locale at runtime
                    "Object.defineProperty(navigator, 'language', {get: ()=>'ko-KR'}); Object.defineProperty(navigator, 'languages', {get: ()=>['ko-KR','ko','en-US','en']});",
                    # Spoof timezone via Intl API
                    "(() => { const _res = Intl.DateTimeFormat.prototype.resolvedOptions; Intl.DateTimeFormat.prototype.resolvedOptions = function(){ const r=_res.apply(this, arguments); return Object.assign(r, { timeZone: 'Asia/Seoul' });}; })();",
                    # Increase touch/hardware hints
                    "try { Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 5}); } catch(e){}",
                ],
                "ch_headers": {
                    "sec-ch-ua-mobile": "?1",
                    "sec-ch-ua-platform": '"Android"',
                }
            }
        return overrides
    @staticmethod
    def get_playwright_launch_args() -> Dict:
        """Get optimized Playwright launch arguments with maximum stealth."""
        user_agent = BrowserConfig.get_random_user_agent()
        return {
            "headless": False,  # Many sites detect headless mode
            "args": BrowserConfig.get_browser_args(headless=False, browser_type="chrome"),
            "ignoreDefaultArgs": [
                "--enable-automation",
                "--enable-blink-features=AutomationControlled",
                "--disable-background-networking",
                "--disable-component-extensions-with-background-pages",
                "--disable-extensions",
                "--disable-sync",
            ],
            "viewport": {"width": 1920, "height": 1080},
            "userAgent": user_agent,
            "bypassCSP": True,
            "ignoreHTTPSErrors": True,
            "javaScriptEnabled": True,
            "acceptDownloads": False,
            "hasTouch": False,
            "isMobile": False,
            "locale": "en-US",
            "timezoneId": "America/New_York",
            "permissions": ["geolocation", "notifications"],
            "deviceScaleFactor": 1,
            "colorScheme": "light",
            "reducedMotion": "no-preference",
            "forcedColors": "none",
            "extraHTTPHeaders": BrowserConfig.get_stealth_headers(),
            # Additional Playwright stealth options
            "chromiumSandbox": False,
            "devtools": False,
        }
    
    @staticmethod
    def get_stealth_scripts() -> List[str]:
        """
        Get JavaScript to inject for additional stealth.
        
        Returns:
            List of JavaScript code strings to inject
        """
        return [
            # Override navigator.webdriver
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """,
            
            # Add chrome object if missing
            """
            if (!window.chrome) {
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
            }
            """,
            
            # Fix permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """,
            
            # Hide automation indicators
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            """,
            
            # Set realistic window properties
            """
            Object.defineProperty(screen, 'availWidth', {
                get: () => 1920
            });
            Object.defineProperty(screen, 'availHeight', {
                get: () => 1040
            });
            """,
            
            # Override language if needed
            """
            Object.defineProperty(navigator, 'language', {
                get: () => 'en-US'
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            """,
        ]