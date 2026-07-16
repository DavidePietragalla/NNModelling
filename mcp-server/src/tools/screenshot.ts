/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 */

import { z } from "zod";
import { captureChromiumScreenshot } from "../chromium-screenshot.js";
import type { ServerContext } from "../server.js";

export const capture_screenshot = {
  schema: z.object({
    outputPath: z.string().min(1).optional(),
    pageUrl: z.string().url().optional(),
    fullPage: z.boolean().optional(),
  }),

  async handler(
    _ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ) {
    return captureChromiumScreenshot(input);
  },
};
