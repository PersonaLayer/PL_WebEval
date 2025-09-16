import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd, cwd: Path, allow_fail: bool = False) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), shell=False)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} in {cwd}")
    return proc.returncode


def _find_cli(vendored_dir: Path) -> Optional[Path]:
    candidates = [
        vendored_dir / "cli.js",
        vendored_dir / "dist" / "cli.js",
        vendored_dir / "build" / "cli.js",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback search
    for p in vendored_dir.rglob("cli.js"):
        return p
    return None


def install_vendored_server(verbose: bool = True) -> Optional[Path]:
    """
    Install Node dependencies and Playwright browsers for the vendored MCP server.
    Returns the resolved cli.js path if successful, otherwise None.
    """
    script_dir = Path(__file__).resolve().parent
    vendored_dir = script_dir / "playwright-mcp-main"
    if not vendored_dir.exists():
        if verbose:
            print(f"[setup_mcp] Vendored directory not found at: {vendored_dir}")
        return None

    pkg_json = vendored_dir / "package.json"
    if not pkg_json.exists():
        if verbose:
            print(f"[setup_mcp] package.json not found in vendored directory: {vendored_dir}")
        return None

    # Ensure node/npm available
    npm = shutil.which("npm")
    npx = shutil.which("npx")
    if not npm or not npx:
        raise EnvironmentError("Node.js tooling not found on PATH. Install Node.js (includes npm).")

    # npm ci or npm install
    lock_file = vendored_dir / "package-lock.json"
    try:
        if lock_file.exists():
            if verbose: print("[setup_mcp] Running: npm ci")
            _run([npm, "ci"], vendored_dir)
        else:
            if verbose: print("[setup_mcp] Running: npm install")
            _run([npm, "install"], vendored_dir)
    except RuntimeError as e:
        if verbose: print(f"[setup_mcp] npm install failed: {e}")
        return None

    # Build if script exists
    try:
        # 'npm run build' may or may not exist; allow failure if script missing
        if verbose: print("[setup_mcp] Attempting: npm run build")
        _run([npm, "run", "build"], vendored_dir, allow_fail=True)
    except Exception:
        pass

    # Install browsers; allow fail on platforms without install-deps
    try:
        if verbose: print("[setup_mcp] Installing Playwright chromium")
        _run([npx, "playwright", "install", "chromium"], vendored_dir, allow_fail=True)
        # Non-Windows often benefits from system deps
        if os.name != "nt":
            if verbose: print("[setup_mcp] Installing Playwright system dependencies")
            _run([npx, "playwright", "install-deps"], vendored_dir, allow_fail=True)
    except Exception:
        pass

    cli = _find_cli(vendored_dir)
    if cli and verbose:
        print(f"[setup_mcp] Detected CLI: {cli}")
    return cli


def main() -> int:
    try:
        cli = install_vendored_server(verbose=True)
        if not cli:
            print("[setup_mcp] Installation completed with warnings; CLI not found.")
            return 1
        print(f"[setup_mcp] Success. CLI at: {cli}")
        return 0
    except Exception as e:
        print(f"[setup_mcp] Error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())