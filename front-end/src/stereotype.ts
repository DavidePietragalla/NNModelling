import type { ShapeDimPattern, ShapePattern, TypeSignature } from './conversion/tensortypes';

// 1. Definiamo i tipi esatti del tuo JSON originale
export interface ModuleParameter {
  type: string;
  default: string;
  position?: "top" | "bottom"
}

export interface StereotypeView {
  color: string;
  width: number;
  height: number;
}

export interface StereotypeJson {
  category?: string;
  pythonClassName?: string;
  taskType?: "classification" | "regression";
  expr?: string;
  view?: Partial<StereotypeView>;
  params?: Record<string, ModuleParameter>;
  type_signature?: TypeSignature;
}

export class Stereotype {
  // Usiamo 'readonly' così siamo sicuri di non sovrascrivere accidentalmente i dati a runtime
  public readonly id: string;
  public readonly name: string; // <-- Calcolato una sola volta!
  public readonly category: string;
  public readonly pythonClassName: string;
  public readonly taskType: string;
  public readonly expr: string;
  public readonly parameters: Record<string, ModuleParameter>;
  public readonly view: StereotypeView;
  public readonly isJoin: boolean;
  public readonly isInput: boolean;
  public readonly isLoss: boolean;
  public readonly isSubFlow: boolean;

  /** Optional type signature for static tensor type checking.
   *  undefined means this stereotype has not been annotated yet. */
  public readonly typeSignature: TypeSignature | undefined;

  constructor(filePath: string, data: StereotypeJson) {
    this.id = filePath;

    // Calcoliamo il nome subito e lo salviamo nella proprietà. 
    // Molto più efficiente di un metodo getName() chiamato in un ciclo #each!
    const fileName = filePath.split(/[\\/]/).pop() || '';
    const lastDot = fileName.lastIndexOf('.');
    this.name = lastDot > 0 ? fileName.substring(0, lastDot) : fileName;

    // Fallback di sicurezza se il JSON è incompleto
    this.category = data.category || "Uncategorized";
    this.pythonClassName = data.pythonClassName || "";
    this.taskType = data.taskType || "";
    this.expr = data.expr || "";
    this.parameters = data.params || {};

    this.isJoin = data.category === "Join" || filePath.includes("/Joins/");
    // Un Input si chiama "Input" nella categoria
    this.isInput = data.category === "Input";

    // Una Loss ha "Loss" nel nome della categoria (es. BCEWithLogitsLoss)
    this.isLoss = data.category === "Loss";
    this.isSubFlow = data.category === "Subflow" || filePath.includes("/SubFlows/");
    // Gestione sicura della view
    this.view = {
      color: data.view?.color || "#4779c4",
      width: data.view?.width || 140,
      height: data.view?.height || 60
    };

    this.typeSignature = this.parseTypeSignature(data.type_signature);
  }

  /**
   * Parse a raw TypeSignature from JSON, stripping the `$` prefix from
   * symbolic dimension names and deep-cloning to avoid mutation.
   * Returns undefined when no signature is present.
   */
  private parseTypeSignature(raw: TypeSignature | undefined): TypeSignature | undefined {
    if (!raw) return undefined;

    const stripPattern = (pattern: ShapePattern): ShapePattern =>
      pattern.map((dim) => this.stripDollar(dim));

    // input is ShapePattern (ShapeDimPattern[]) for modules/subflows,
    // or ShapePattern[] (ShapeDimPattern[][]) for joins.
    // Distinguish by checking whether the first element is itself an array.
    let input: ShapePattern | ShapePattern[];
    if (Array.isArray(raw.input) && raw.input.length > 0 && Array.isArray(raw.input[0])) {
      input = (raw.input as ShapePattern[]).map((p) => stripPattern(p));
    } else {
      input = stripPattern(raw.input as ShapePattern);
    }

    return {
      kind: raw.kind,
      input,
      output: stripPattern(raw.output),
      dtype: raw.dtype ? { ...raw.dtype } : undefined,
    };
  }

  /** Deep‑clone a single dimension pattern, stripping the leading `$` from symbolic names. */
  private stripDollar(pattern: ShapeDimPattern): ShapeDimPattern {
    if (pattern.kind === 'symbolic' && pattern.name.startsWith('$')) {
      return { ...pattern, name: pattern.name.slice(1) };
    }
    return pattern;
  }

  // ---- IL CARICATORE AUTOMATICO ----
  // Questo metodo usa le API di Vite per leggere tutti i file JSON in una cartella
  public static loadFromDirectory(): Stereotype[] {
    // import.meta.glob è magico in Vite: trova i file a compile-time.
    // Assicurati che il path relativo parta da dove si trova QUESTO file .ts
    // (Adegua il path se la tua cartella Modules si trova altrove)
    const files = import.meta.glob('../../Stereotypes/**/*.json', { eager: true });

    const loadedStereotypes: Stereotype[] = [];

    for (const [path, rawData] of Object.entries(files)) {
      // Vite impacchetta il JSON importato in un oggetto con la proprietà 'default'
      const jsonData = (rawData as any).default || rawData;

      try {
        loadedStereotypes.push(new Stereotype(path, jsonData));
      } catch (e) {
        console.error(`Errore nel caricamento dello stereotipo da ${path}:`, e);
      }
    }

    // Opzionale: li ordiniamo alfabeticamente per averli belli in ordine nel Dropdown
    return loadedStereotypes.sort((a, b) => a.name.localeCompare(b.name));
  }
}
