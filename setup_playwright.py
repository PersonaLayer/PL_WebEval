#!/usr/bin/env python3
"""
Setup script for Playwright MCP server for PL_WebEval.
This ensures the browser automation component is properly installed.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_node():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js is not installed!")
    print("Please install Node.js from https://nodejs.org/")
    print("Recommended: Node.js 18.x or later")
    return False

def install_playwright_mcp():
    """Install the Playwright MCP server globally."""
    print("\n📦 Installing Playwright MCP server...")
    
    try:
        # Install the Playwright MCP server globally
        cmd = ["npm", "install", "-g", "@upstash/playwright-mcp-server"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ npm install had issues: {result.stderr}")
            # Try with npx instead
            print("\n🔄 Trying alternative installation with npx...")
            cmd = ["npx", "-y", "@upstash/playwright-mcp-server", "--version"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Playwright MCP server is available via npx")
                return True
        else:
            print("✅ Playwright MCP server installed successfully")
            return True
            
    except FileNotFoundError:
        print("❌ npm is not found. Please ensure Node.js is properly installed.")
        return False
    except Exception as e:
        print(f"❌ Error installing Playwright MCP: {e}")
        return False

def install_playwright_browsers():
    """Install Playwright browsers."""
    print("\n🌐 Installing Playwright browsers...")
    
    try:
        # Install Chromium browser for Playwright
        cmd = ["npx", "playwright", "install", "chromium"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Chromium browser installed")
        else:
            print(f"⚠️ Browser installation had issues: {result.stderr}")
            
        # Install system dependencies on Linux
        if sys.platform.startswith('linux'):
            print("\n📦 Installing system dependencies (Linux)...")
            cmd = ["npx", "playwright", "install-deps", "chromium"]
            subprocess.run(cmd, capture_output=True, text=True)
            print("✅ System dependencies installed")
            
    except Exception as e:
        print(f"⚠️ Could not install browsers: {e}")
        print("You may need to run: npx playwright install chromium")

def create_env_file():
    """Create or update .env file with Playwright settings."""
    env_path = Path("PersonaLayer_Main/PL_WebEval/.env")
    
    print(f"\n📝 Configuring environment settings...")
    
    env_content = []
    playwright_configured = False
    
    # Read existing .env if it exists
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith("PLAYWRIGHT_"):
                    playwright_configured = True
                env_content.append(line.rstrip())
    
    # Add Playwright configuration if not present
    if not playwright_configured:
        env_content.extend([
            "",
            "# Playwright MCP Configuration",
            "# Set to 'true' to run browser in headless mode (no visible window)",
            "# Set to 'false' for better compatibility with bot detection",
            "PLAYWRIGHT_HEADLESS=false",
            "",
            "# Optional: Path to custom Playwright MCP CLI",
            "# PLAYWRIGHT_MCP_CLI_PATH=/path/to/playwright-mcp/cli.js",
        ])
        
        with open(env_path, 'w') as f:
            f.write('\n'.join(env_content))
        
        print(f"✅ Updated .env file with Playwright settings")
    else:
        print(f"✅ Playwright settings already configured in .env")

def test_playwright_connection():
    """Test if Playwright can be launched."""
    print("\n🧪 Testing Playwright connection...")
    
    try:
        test_script = """
const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto('https://example.com');
    const title = await page.title();
    console.log('Page title:', title);
    await browser.close();
    console.log('SUCCESS');
})();
"""
        result = subprocess.run(
            ["node", "-e", test_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "SUCCESS" in result.stdout:
            print("✅ Playwright is working correctly!")
            return True
        else:
            print(f"⚠️ Playwright test had issues: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Playwright test timed out")
        return False
    except Exception as e:
        print(f"⚠️ Could not test Playwright: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🔧 PL_WebEval Playwright Setup")
    print("=" * 60)
    
    # Step 1: Check Node.js
    if not check_node():
        print("\n❌ Setup failed: Node.js is required")
        print("\nPlease install Node.js first:")
        print("1. Visit https://nodejs.org/")
        print("2. Download and install Node.js (LTS version recommended)")
        print("3. Run this setup script again")
        return 1
    
    # Step 2: Install Playwright MCP
    if not install_playwright_mcp():
        print("\n⚠️ Playwright MCP installation had issues")
        print("\nYou can try manual installation:")
        print("1. Run: npm install -g @upstash/playwright-mcp-server")
        print("2. Or use: npx -y @upstash/playwright-mcp-server")
    
    # Step 3: Install browsers
    install_playwright_browsers()
    
    # Step 4: Configure environment
    create_env_file()
    
    # Step 5: Test connection
    if test_playwright_connection():
        print("\n" + "=" * 60)
        print("✅ Setup completed successfully!")
        print("=" * 60)
        print("\n📌 Next steps:")
        print("1. Ensure your OPENROUTER_API_KEY is set in the .env file")
        print("2. Run: python -m pl_webeval.cli")
        print("\n💡 Tips:")
        print("- For better bot evasion, keep PLAYWRIGHT_HEADLESS=false")
        print("- If you encounter issues, try running with visible browser")
        print("- Some sites may still block automation - this is expected")
        return 0
    else:
        print("\n⚠️ Setup completed with warnings")
        print("\nIf Playwright isn't working, try:")
        print("1. Run: npx playwright install chromium")
        print("2. On Linux: sudo npx playwright install-deps")
        print("3. Check firewall/antivirus settings")
        return 1

if __name__ == "__main__":
    sys.exit(main())