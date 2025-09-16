# Vendoring the Custom Playwright MCP Server (inject-tools)

Context
- This project relies on a customized Playwright MCP server (non-official) extended with inject-tools (see your docs/inject-tools.md in the custom repo).
- We ship this customized server inside the Python package for reproducibility. At runtime, PL-WebEval will prefer an explicit PLAYWRIGHT_MCP_CLI_PATH but will also fall back to the vendored CLI under src/pl_webeval/playwright-mcp-main/cli.js if present.

Directory layout (after vendoring)
- src/pl_webeval/playwright-mcp-main/ ... (copied from your custom repo, excluding node_modules/.git/etc.)

One-time setup (Windows PowerShell)
1) Place/confirm your customized server at a source path (example this repo layout):
   - ModularV2/playwright-mcp-main (contains package.json, cli.js or dist/cli.js, your inject-tools, docs/inject-tools.md, etc.)
2) From PersonaLayer_Main/PL_WebEval, run the helper to vendor and build:
   - .\scripts\vendor_playwright_mcp.ps1 -SourcePath "..\..\ModularV2\playwright-mcp-main"
   - What it does:
     - Copies source → src/pl_webeval/playwright-mcp-main (excluding node_modules, .git, dist, etc.)
     - Runs npm ci (or npm install) and npm run build if available
     - Locates cli.js and writes PLAYWRIGHT_MCP_CLI_PATH into .env
3) Verify .env contains:
   - OPENROUTER_API_KEY=... (your key)
   - PLAYWRIGHT_MCP_CLI_PATH=...absolute path to cli.js inside src/pl_webeval/playwright-mcp-main/...

Re-running after changes
- If you edit your custom MCP server (e.g., inject-tools), re-run:
  - .\scripts\vendor_playwright_mcp.ps1 -SourcePath "..\..\ModularV2\playwright-mcp-main"

Runtime resolution order (inside PL-WebEval)
- First: PLAYWRIGHT_MCP_CLI_PATH (env) if defined
- Fallback: src/pl_webeval/playwright-mcp-main/cli.js (vendored)
- If neither exists, PL-WebEval prints an error and stops.

Packaging and distribution
- setup (pyproject.toml) includes package-data to ship src/pl_webeval/playwright-mcp-main/** in wheels.
- MANIFEST.in includes the vendored server for sdists and excludes heavyweight folders (node_modules, .git, etc.).
- Build:
  - python -m pip install --upgrade build
  - python -m build
  - Verify the generated wheel/sdist contains src/pl_webeval/playwright-mcp-main/ (no node_modules).
- Install the package artifact and run:
  - pip install dist/pl_webeval-*.whl
  - Set OPENROUTER_API_KEY and optionally PLAYWRIGHT_MCP_CLI_PATH (or rely on vendored fallback)
  - python -m pl_webeval.cli --testcases data/test_cases.csv --rundir out/run_vendor

Licensing and notices [VERIFY]
- Ensure the custom server's LICENSE is present in the vendored copy (src/pl_webeval/playwright-mcp-main/).
- Maintain THIRD_PARTY_NOTICES.md at the project root listing the vendored server’s license and upstream URL.
- Confirm any upstream requirements (NOTICE files, headers) are preserved.

Docs linkage
- For details of the inject-tools, consult your original repository's docs/inject-tools.md. That file is copied during vendoring (unless excluded) and will reside under src/pl_webeval/playwright-mcp-main/docs/inject-tools.md.

Troubleshooting
- Node not found: Install Node.js and ensure node/npm are on PATH.
- CLI not found after build: Confirm your build emits cli.js (or update vendor script to point to the correct entry).
- Provider rejects inline images: The LLM client automatically retries without image parts for providers with inline_data restrictions.