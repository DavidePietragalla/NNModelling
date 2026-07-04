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
