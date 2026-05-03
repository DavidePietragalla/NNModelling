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
  expr?: string;
  view?: Partial<StereotypeView>;
  params?: Record<string, ModuleParameter>;
}

export class Stereotype {
  // Usiamo 'readonly' così siamo sicuri di non sovrascrivere accidentalmente i dati a runtime
  public readonly id: string;
  public readonly name: string; // <-- Calcolato una sola volta!
  public readonly category: string;
  public readonly pythonClassName: string;
  public readonly expr: string;
  public readonly parameters: Record<string, ModuleParameter>;
  public readonly view: StereotypeView;
  public readonly isJoin: boolean;
  public readonly isInput: boolean;
  public readonly isLoss: boolean;

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
    this.expr = data.expr || "";
    this.parameters = data.params || {};

    this.isJoin = data.category === "Join" || filePath.includes("/Joins/");
    // Un Input si chiama "Input" nella categoria
    this.isInput = data.category === "Input";

    // Una Loss ha "Loss" nel nome della categoria (es. BCEWithLogitsLoss)
    this.isLoss = data.category?.includes("Loss") || false;
    // Gestione sicura della view
    this.view = {
      color: data.view?.color || "#4779c4",
      width: data.view?.width || 140,
      height: data.view?.height || 60
    };
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
