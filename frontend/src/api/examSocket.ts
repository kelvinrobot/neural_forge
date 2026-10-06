import type { ControlAction, MessageOf, ServerMessage, ServerMessageType } from "./types";

type Handler<T extends ServerMessageType> = (message: MessageOf<T>) => void;
type AnyHandler = (message: ServerMessage) => void;

export type SocketState = "idle" | "connecting" | "open" | "closed" | "lost";

const PING_INTERVAL_MS = 20_000;
const BACKPRESSURE_LIMIT_BYTES = 1_000_000;

/** One WebSocket per examination. It never reconnects by itself; an unexpected close is reported as "lost". */
export class ExamSocket {
  private socket: WebSocket | null = null;
  private pingTimer: number | null = null;
  private closedByClient = false;
  private readonly handlers = new Map<ServerMessageType, Set<AnyHandler>>();
  private readonly anyHandlers = new Set<AnyHandler>();
  private readonly stateHandlers = new Set<(state: SocketState) => void>();
  private state: SocketState = "idle";

  constructor(private readonly url: string) {}

  get currentState(): SocketState {
    return this.state;
  }

  connect(): Promise<void> {
    this.setState("connecting");
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      this.socket = socket;
      let opened = false;
      socket.onopen = () => {
        opened = true;
        this.setState("open");
        this.pingTimer = window.setInterval(() => this.send({ type: "ping" }), PING_INTERVAL_MS);
        resolve();
      };
      socket.onmessage = (event) => this.dispatch(event.data);
      socket.onerror = () => {
        if (!opened) reject(new Error("The examination connection could not be opened."));
      };
      socket.onclose = () => {
        this.clearPing();
        this.setState(this.closedByClient ? "closed" : "lost");
        if (!opened) reject(new Error("The examination connection was refused."));
      };
    });
  }

  on<T extends ServerMessageType>(type: T, handler: Handler<T>): () => void {
    const set = this.handlers.get(type) ?? new Set<AnyHandler>();
    set.add(handler as AnyHandler);
    this.handlers.set(type, set);
    return () => set.delete(handler as AnyHandler);
  }

  onAny(handler: AnyHandler): () => void {
    this.anyHandlers.add(handler);
    return () => this.anyHandlers.delete(handler);
  }

  onState(handler: (state: SocketState) => void): () => void {
    this.stateHandlers.add(handler);
    return () => this.stateHandlers.delete(handler);
  }

  /** Returns false when the frame was dropped for backpressure or because the socket is not open. */
  sendFrame(image: string): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    if (this.socket.bufferedAmount > BACKPRESSURE_LIMIT_BYTES) return false;
    this.socket.send(JSON.stringify({ type: "frame", timestamp_ms: performance.now(), image }));
    return true;
  }

  sendControl(action: ControlAction): void {
    this.send({ type: "control", action });
  }

  close(): void {
    this.closedByClient = true;
    this.clearPing();
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) {
      this.socket.close(1000, "client closed");
    }
    this.socket = null;
    this.handlers.clear();
    this.anyHandlers.clear();
    this.stateHandlers.clear();
  }

  private send(payload: object): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  private dispatch(raw: unknown): void {
    if (typeof raw !== "string") return;
    let message: ServerMessage;
    try {
      message = JSON.parse(raw) as ServerMessage;
    } catch {
      return;
    }
    if (!message || typeof message.type !== "string") return;
    this.anyHandlers.forEach((handler) => handler(message));
    this.handlers.get(message.type)?.forEach((handler) => handler(message));
  }

  private setState(state: SocketState): void {
    this.state = state;
    this.stateHandlers.forEach((handler) => handler(state));
  }

  private clearPing(): void {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
}
