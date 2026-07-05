/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

// front-end/src/stereotype.ts
import { StereotypeCore, type StereotypeJson } from "./core/StereotypeCore";

// Re-export interfaces for backward compatibility
export type { ModuleParameter, StereotypeView, StereotypeJson } from "./core/StereotypeCore";

export class Stereotype extends StereotypeCore {
  constructor(filePath: string, data: StereotypeJson) {
    super(filePath, data);
  }

  // Override with Vite-specific loader
  public static loadFromDirectory(): Stereotype[] {
    // Delegate to StereotypeCore's Vite loader, cast to Stereotype[]
    return StereotypeCore.loadFromDirectory() as Stereotype[];
  }
}
