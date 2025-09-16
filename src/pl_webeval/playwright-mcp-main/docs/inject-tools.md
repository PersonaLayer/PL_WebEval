# CSS and JavaScript Injection Tools for Playwright MCP Server

This document describes the new CSS and JavaScript injection tools added to the Playwright MCP server.

## Overview

Two new tools have been added to enable dynamic modification of web pages:

1. **browser_inject_css** - Inject custom CSS styles into the current page
2. **browser_inject_js** - Inject and execute JavaScript code in the page context

## Tool Descriptions

### browser_inject_css

Injects custom CSS styles into the current page to modify its appearance.

**Parameters:**
- `css` (string, required): The CSS code to inject into the page

**Example Usage:**
```json
{
  "tool": "browser_inject_css",
  "arguments": {
    "css": "body { background-color: #f0f0f0; } h1 { color: red; }"
  }
}
```

### browser_inject_js

Injects and executes custom JavaScript code in the current page context.

**Parameters:**
- `javascript` (string, required): The JavaScript code to inject and execute in the page

**Example Usage:**
```json
{
  "tool": "browser_inject_js",
  "arguments": {
    "javascript": "document.querySelector('h1').textContent = 'Modified Title'; return document.title;"
  }
}
```

## Usage Examples

### Example 1: Styling a webpage

```javascript
// Navigate to a page
await mcp.callTool('browser_navigate', { url: 'https://example.com' });

// Inject CSS to style the page
await mcp.callTool('browser_inject_css', {
  css: `
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(to right, #667eea, #764ba2);
      color: white;
    }
    
    h1 {
      text-align: center;
      font-size: 3em;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    a {
      color: #ffd700;
      text-decoration: none;
    }
    
    a:hover {
      text-decoration: underline;
    }
  `
});
```

### Example 2: Modifying page content with JavaScript

```javascript
// Navigate to a page
await mcp.callTool('browser_navigate', { url: 'https://example.com' });

// Inject JavaScript to modify the page
const result = await mcp.callTool('browser_inject_js', {
  javascript: `
    // Change all paragraph text
    const paragraphs = document.querySelectorAll('p');
    paragraphs.forEach((p, index) => {
      p.textContent = 'Modified paragraph ' + (index + 1);
    });
    
    // Add a new element
    const newDiv = document.createElement('div');
    newDiv.style.cssText = 'position: fixed; top: 10px; right: 10px; background: yellow; padding: 10px; border: 2px solid black;';
    newDiv.textContent = 'Injected by MCP!';
    document.body.appendChild(newDiv);
    
    // Return some information
    return {
      paragraphCount: paragraphs.length,
      title: document.title,
      url: window.location.href
    };
  `
});
```

### Example 3: Dark mode toggle

```javascript
// Inject CSS for dark mode
await mcp.callTool('browser_inject_css', {
  css: `
    .dark-mode {
      background-color: #1a1a1a !important;
      color: #e0e0e0 !important;
    }
    
    .dark-mode * {
      background-color: #2a2a2a !important;
      color: #e0e0e0 !important;
      border-color: #444 !important;
    }
    
    .dark-mode a {
      color: #66b3ff !important;
    }
    
    .dark-mode img {
      opacity: 0.8;
    }
  `
});

// Toggle dark mode with JavaScript
await mcp.callTool('browser_inject_js', {
  javascript: `
    document.body.classList.toggle('dark-mode');
    return document.body.classList.contains('dark-mode') ? 'Dark mode enabled' : 'Dark mode disabled';
  `
});
```

### Example 4: Form auto-fill

```javascript
// Auto-fill a form using JavaScript injection
await mcp.callTool('browser_inject_js', {
  javascript: `
    // Find and fill form inputs
    const inputs = {
      '#username': 'testuser',
      '#email': 'test@example.com',
      '#phone': '555-1234',
      '#message': 'This is an automated test message'
    };
    
    let filled = 0;
    for (const [selector, value] of Object.entries(inputs)) {
      const element = document.querySelector(selector);
      if (element) {
        element.value = value;
        element.dispatchEvent(new Event('input', { bubbles: true }));
        filled++;
      }
    }
    
    return { filledFields: filled, totalFields: Object.keys(inputs).length };
  `
});
```

### Example 5: Remove ads and popups

```javascript
// Remove common ad elements
await mcp.callTool('browser_inject_css', {
  css: `
    /* Hide common ad containers */
    [class*="ad-"], [id*="ad-"],
    [class*="advertisement"], [id*="advertisement"],
    [class*="banner"], [id*="banner"],
    .popup, .modal, .overlay {
      display: none !important;
    }
  `
});

// Remove popups with JavaScript
await mcp.callTool('browser_inject_js', {
  javascript: `
    // Remove popup elements
    const popupSelectors = ['.popup', '.modal', '.overlay', '[class*="popup"]', '[id*="popup"]'];
    let removed = 0;
    
    popupSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => {
        el.remove();
        removed++;
      });
    });
    
    // Disable popup triggers
    window.addEventListener('click', (e) => e.stopPropagation(), true);
    
    return { removedElements: removed };
  `
});
```

## Integration with MCP Clients

MCP clients can now use these tools to:

1. **Customize webpage appearance** - Apply custom themes, fix layout issues, or improve readability
2. **Automate interactions** - Fill forms, click buttons, or extract data programmatically
3. **Debug and test** - Inject test scripts, monitor page behavior, or simulate user actions
4. **Enhance accessibility** - Add screen reader support, increase contrast, or modify fonts
5. **Remove distractions** - Hide ads, popups, or unnecessary elements

## Error Handling

The `browser_inject_js` tool includes error handling:
- If the JavaScript code throws an error, it will be caught and returned in the response
- The result will indicate whether the execution was successful or if an error occurred
- Any return value from the JavaScript code will be serialized and included in the response

## Security Considerations

These tools execute code in the context of the current page:
- CSS injection is generally safe but can affect page layout and appearance
- JavaScript injection has full access to the page's DOM and can modify any content
- Use these tools responsibly and only on pages where you have permission to modify content
- Be cautious when injecting code from untrusted sources

## Generated Playwright Code

Both tools generate corresponding Playwright code that can be used in test scripts:

For CSS injection:
```javascript
await page.addStyleTag({ content: `/* your CSS here */` });
```

For JavaScript injection:
```javascript
await page.evaluate((code) => {
  const fn = new Function(code);
  return fn();
}, `/* your JavaScript here */`);
