export interface ConnectionStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface SavedBackendConnection {
  version: 1;
  baseUrl: string;
  token: string;
  connectionId: string;
  requestId: string | null;
  verificationCode: string | null;
  deviceName: string | null;
}

interface PersistedConnections {
  version: 1;
  activeUrl: string | null;
  connections: Record<string, SavedBackendConnection>;
}

const STORAGE_KEY = "nnm.training.connections";

export function normalizeBackendUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error("Inserisci l'URL del backend");
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error("L'URL del backend deve essere assoluto");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Il backend deve usare HTTP o HTTPS");
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("L'URL del backend non può contenere credenziali, query o frammenti");
  }
  const path = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${path}`;
}

export function loadBackendConnection(storage: ConnectionStorage = browserStorage()): SavedBackendConnection | null {
  const state = readState(storage);
  if (!state.activeUrl) return null;
  return state.connections[state.activeUrl] ?? null;
}

export function saveBackendConnection(
  connection: SavedBackendConnection,
  storage: ConnectionStorage = browserStorage(),
): void {
  const state = readState(storage);
  state.connections[connection.baseUrl] = connection;
  state.activeUrl = connection.baseUrl;
  storage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function forgetBackendConnection(
  baseUrl: string,
  storage: ConnectionStorage = browserStorage(),
): void {
  const state = readState(storage);
  delete state.connections[baseUrl];
  if (state.activeUrl === baseUrl) {
    state.activeUrl = Object.keys(state.connections).at(-1) ?? null;
  }
  if (Object.keys(state.connections).length === 0) {
    storage.removeItem(STORAGE_KEY);
  } else {
    storage.setItem(STORAGE_KEY, JSON.stringify(state));
  }
}

function readState(storage: ConnectionStorage): PersistedConnections {
  const empty: PersistedConnections = { version: 1, activeUrl: null, connections: {} };
  const raw = storage.getItem(STORAGE_KEY);
  if (!raw) return empty;
  try {
    const parsed = JSON.parse(raw) as Partial<PersistedConnections>;
    if (parsed.version !== 1 || typeof parsed.connections !== "object" || parsed.connections === null) {
      return empty;
    }
    const connections = Object.fromEntries(
      Object.entries(parsed.connections).filter((entry): entry is [string, SavedBackendConnection] => {
        const value = entry[1] as Partial<SavedBackendConnection>;
        return value.version === 1 && typeof value.baseUrl === "string" && typeof value.token === "string";
      }),
    );
    return {
      version: 1,
      activeUrl: typeof parsed.activeUrl === "string" ? parsed.activeUrl : null,
      connections,
    };
  } catch {
    return empty;
  }
}

function browserStorage(): ConnectionStorage {
  if (!("localStorage" in globalThis)) {
    throw new Error("localStorage non è disponibile");
  }
  return globalThis.localStorage;
}
