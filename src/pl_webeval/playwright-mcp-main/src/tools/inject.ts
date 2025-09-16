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

import { z } from 'zod';
import { defineTool, type ToolFactory } from './tool.js';

const injectCSS: ToolFactory = captureSnapshot => defineTool({
  capability: 'core',

  schema: {
    name: 'browser_inject_css',
    title: 'Inject CSS into page',
    description: 'Inject custom CSS styles into the current page to modify its appearance',
    inputSchema: z.object({
      css: z.string().describe('The CSS code to inject into the page'),
    }),
    type: 'destructive',
  },

  handle: async (context, params) => {
    const tab = await context.ensureTab();

    // Inject CSS into the page
    await tab.page.addStyleTag({ content: params.css });

    const code = [
      `// Inject CSS into the page`,
      `await page.addStyleTag({ content: \`${params.css.replace(/`/g, '\\`')}\` });`,
    ];

    return {
      code,
      captureSnapshot,
      waitForNetwork: false,
    };
  },
});

const injectJS: ToolFactory = captureSnapshot => defineTool({
  capability: 'core',

  schema: {
    name: 'browser_inject_js',
    title: 'Inject JavaScript into page',
    description: 'Inject and execute custom JavaScript code in the current page context',
    inputSchema: z.object({
      javascript: z.string().describe('The JavaScript code to inject and execute in the page'),
    }),
    type: 'destructive',
  },

  handle: async (context, params) => {
    const tab = await context.ensureTab();

    // Evaluate JavaScript in the page context
    await tab.page.evaluate((code: string) => {
      try {
        // Use Function constructor to evaluate the code and return result
        const fn = new Function(code);
        return fn();
      } catch (error) {
        return { error: error instanceof Error ? error.toString() : String(error) };
      }
    }, params.javascript);

    const code = [
      `// Inject and execute JavaScript in the page`,
      `await page.evaluate((code) => {`,
      `  const fn = new Function(code);`,
      `  return fn();`,
      `}, \`${params.javascript.replace(/`/g, '\\`')}\`);`,
    ];

    return {
      code,
      captureSnapshot,
      waitForNetwork: false,
    };
  },
});

export default (captureSnapshot: boolean) => [
  injectCSS(captureSnapshot),
  injectJS(captureSnapshot),
];
