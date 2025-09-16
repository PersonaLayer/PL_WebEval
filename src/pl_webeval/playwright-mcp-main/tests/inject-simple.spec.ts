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

test('browser_inject_css basic test', async ({ client, server }) => {
  server.setContent('/', `
    <!DOCTYPE html>
    <html>
      <head>
        <title>Test Page</title>
      </head>
      <body>
        <h1>Hello World</h1>
      </body>
    </html>
  `, 'text/html');

  // Navigate to test page
  await client.callTool({
    name: 'browser_navigate',
    arguments: { url: server.PREFIX },
  });

  // Inject CSS
  const result = await client.callTool({
    name: 'browser_inject_css',
    arguments: {
      css: 'body { background-color: red; }',
    },
  });

  // Check that the tool executed without error
  expect(result.isError).toBeFalsy();

  // The result should contain the Playwright code
  expect(result).toContainTextContent('```js');
  expect(result).toContainTextContent('await page.addStyleTag({ content: `body { background-color: red; }` });');
});

test('browser_inject_js basic test', async ({ client, server }) => {
  server.setContent('/', `
    <!DOCTYPE html>
    <html>
      <head>
        <title>Test Page</title>
      </head>
      <body>
        <h1 id="title">Hello World</h1>
      </body>
    </html>
  `, 'text/html');

  // Navigate to test page
  await client.callTool({
    name: 'browser_navigate',
    arguments: { url: server.PREFIX },
  });

  // Inject JavaScript
  const result = await client.callTool({
    name: 'browser_inject_js',
    arguments: {
      javascript: 'return document.getElementById("title").textContent;',
    },
  });

  // Check that the tool executed without error
  expect(result.isError).toBeFalsy();

  // The result should contain the Playwright code
  expect(result).toContainTextContent('```js');
  expect(result).toContainTextContent('await page.evaluate((code) => {');
});
