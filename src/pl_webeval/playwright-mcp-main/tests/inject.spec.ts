/**
 * Copyright (c) Microsoft Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { test, expect } from './fixtures.js';

test.describe('CSS and JavaScript Injection', () => {
  test.beforeEach(async ({ server }) => {
    server.setContent('/', `
      <!DOCTYPE html>
      <html>
        <head>
          <title>Test Page</title>
          <style>
            body { background-color: white; color: black; }
            h1 { color: blue; }
          </style>
        </head>
        <body>
          <h1 id="title">Original Title</h1>
          <p class="content">Original paragraph content</p>
          <div id="container">
            <span>Test content</span>
          </div>
        </body>
      </html>
    `, 'text/html');
  });

  test('browser_inject_css should inject CSS styles', async ({ client, server }) => {
    // Navigate to test page
    await client.callTool({
      name: 'browser_navigate',
      arguments: { url: server.PREFIX },
    });

    // Inject CSS
    const result = await client.callTool({
      name: 'browser_inject_css',
      arguments: {
        css: `
          body {
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
          }
          
          h1 {
            color: red !important;
            font-size: 2em;
          }
          
          .custom-class {
            border: 2px solid green;
            padding: 10px;
          }
        `,
      },
    });

    expect(result.isError).toBeFalsy();

    // The result should contain the Playwright code
    const resultText = (result as any).content[0].text;
    expect(resultText).toContain('```js');
    expect(resultText).toContain('await page.addStyleTag({ content: `');
  });

  test('browser_inject_js should execute JavaScript and return results', async ({ client, server }) => {
    // Navigate to test page
    await client.callTool({
      name: 'browser_navigate',
      arguments: { url: server.PREFIX },
    });

    // Inject JavaScript that modifies the page and returns data
    const result = await client.callTool({
      name: 'browser_inject_js',
      arguments: {
        javascript: `
          // Modify the title
          const title = document.getElementById('title');
          const originalTitle = title.textContent;
          title.textContent = 'Modified Title';
          
          // Add a new element
          const newDiv = document.createElement('div');
          newDiv.id = 'injected-element';
          newDiv.textContent = 'This was injected!';
          newDiv.style.color = 'green';
          document.body.appendChild(newDiv);
          
          // Modify paragraph content
          const paragraph = document.querySelector('.content');
          paragraph.textContent = 'Modified paragraph content';
          
          // Return some data
          return {
            originalTitle: originalTitle,
            newTitle: title.textContent,
            elementCount: document.querySelectorAll('*').length,
            injectedElementExists: document.getElementById('injected-element') !== null
          };
        `,
      },
    });

    expect(result.isError).toBeFalsy();

    // The result should contain the Playwright code
    const resultText = (result as any).content[0].text;
    expect(resultText).toContain('```js');
    expect(resultText).toContain('await page.evaluate((code) => {');
  });

  test('browser_inject_js should handle errors gracefully', async ({ client, server }) => {
    // Navigate to test page
    await client.callTool({
      name: 'browser_navigate',
      arguments: { url: server.PREFIX },
    });

    // Inject JavaScript with an error
    const result = await client.callTool({
      name: 'browser_inject_js',
      arguments: {
        javascript: `
          // This will throw an error
          nonExistentFunction();
          return 'This should not be reached';
        `,
      },
    });

    expect(result.isError).toBeFalsy();

    // The result should contain the Playwright code
    const resultText = (result as any).content[0].text;
    expect(resultText).toContain('```js');
    expect(resultText).toContain('await page.evaluate((code) => {');
  });

  test('browser_inject_css and browser_inject_js should work together', async ({ client, server }) => {
    // Navigate to test page
    await client.callTool({
      name: 'browser_navigate',
      arguments: { url: server.PREFIX },
    });

    // First inject CSS for styling
    const cssResult = await client.callTool({
      name: 'browser_inject_css',
      arguments: {
        css: `
          .highlight {
            background-color: yellow;
            padding: 5px;
            border: 2px solid orange;
          }
          
          .hidden {
            display: none;
          }
        `,
      },
    });

    expect(cssResult.isError).toBeFalsy();

    // Then use JavaScript to apply the CSS classes
    const jsResult = await client.callTool({
      name: 'browser_inject_js',
      arguments: {
        javascript: `
          // Apply highlight class to title
          document.getElementById('title').classList.add('highlight');
          
          // Hide the original paragraph
          document.querySelector('.content').classList.add('hidden');
          
          // Create a new highlighted element
          const newElement = document.createElement('p');
          newElement.textContent = 'This element uses injected CSS';
          newElement.classList.add('highlight');
          document.body.appendChild(newElement);
          
          return {
            highlightedElements: document.querySelectorAll('.highlight').length,
            hiddenElements: document.querySelectorAll('.hidden').length
          };
        `,
      },
    });

    expect(jsResult.isError).toBeFalsy();

    // Both tools should have executed successfully
    const cssText = (cssResult as any).content[0].text;
    const jsText = (jsResult as any).content[0].text;

    expect(cssText).toContain('```js');
    expect(cssText).toContain('await page.addStyleTag({ content: `');

    expect(jsText).toContain('```js');
    expect(jsText).toContain('await page.evaluate((code) => {');
  });

  test('browser_inject_js should handle complex return values', async ({ client, server }) => {
    // Navigate to test page
    await client.callTool({
      name: 'browser_navigate',
      arguments: { url: server.PREFIX },
    });

    // Inject JavaScript that returns various data types
    const result = await client.callTool({
      name: 'browser_inject_js',
      arguments: {
        javascript: `
          // Return complex data structure
          return {
            string: 'Hello World',
            number: 42,
            boolean: true,
            array: [1, 2, 3, 'four'],
            nested: {
              level1: {
                level2: {
                  value: 'deeply nested'
                }
              }
            },
            null: null,
            undefined: undefined,
            date: new Date('2024-01-01').toISOString(),
            elements: Array.from(document.querySelectorAll('*')).map(el => el.tagName)
          };
        `,
      },
    });

    expect(result.isError).toBeFalsy();

    // The result should contain the Playwright code
    const resultText = (result as any).content[0].text;
    expect(resultText).toContain('```js');
    expect(resultText).toContain('await page.evaluate((code) => {');
  });
});
