/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
