/**
 * useGlossWS — WebSocket 기반 글로스 스트리밍 훅
 * pause / resume 지원
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { MotionClip } from "../api/gloss";

export type WSStatus = "connecting" | "idle" | "streaming" | "done" | "error" | "closed";

export interface GlossWSState {
  status: WSStatus;
  currentClip: MotionClip | null;
  currentIndex: number;
  total: number;
  tokens: string[];
  errorMsg: string;
  paused: boolean;
}

const INITIAL: GlossWSState = {
  status: "connecting",
  currentClip: null,
  currentIndex: -1,
  total: 0,
  tokens: [],
  errorMsg: "",
  paused: false,
};

export function useGlossWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<GlossWSState>(INITIAL);

  const WS_URL = `ws://${window.location.host}/ws/gloss`;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setState((s) => ({ ...s, status: "connecting", paused: false }));
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setState((s) => ({ ...s, status: "idle" }));

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data as string);
      switch (msg.type) {
        case "start":
          setState((s) => ({
            ...s,
            status: "streaming",
            total: msg.total,
            tokens: msg.tokens,
            currentIndex: -1,
            currentClip: null,
            paused: false,
          }));
          break;
        case "clip":
          setState((s) => ({
            ...s,
            currentClip: msg.clip as MotionClip,
            currentIndex: msg.index as number,
          }));
          break;
        case "done":
          setState((s) => ({ ...s, status: "done", currentClip: null, paused: false }));
          break;
        case "paused":
          setState((s) => ({ ...s, paused: true }));
          break;
        case "resumed":
          setState((s) => ({ ...s, paused: false }));
          break;
        case "error":
          setState((s) => ({ ...s, status: "error", errorMsg: msg.message }));
          break;
      }
    };

    ws.onerror = () =>
      setState((s) => ({ ...s, status: "error", errorMsg: "WebSocket 연결 오류" }));

    ws.onclose = () => setState((s) => ({ ...s, status: "closed" }));
  }, [WS_URL]);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);

  const sendText = useCallback((text: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setState((s) => ({ ...s, status: "error", errorMsg: "WS 연결 안됨" }));
      return;
    }
    setState((s) => ({
      ...s,
      status: "streaming",
      currentIndex: -1,
      currentClip: null,
      tokens: [],
      errorMsg: "",
      paused: false,
    }));
    ws.send(JSON.stringify({ text }));
  }, []);

  const pause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "pause" }));
  }, []);

  const resume = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "resume" }));
  }, []);

  const reset = useCallback(() => {
    setState((s) => ({
      ...s,
      status: "idle",
      currentClip: null,
      currentIndex: -1,
      tokens: [],
      paused: false,
    }));
  }, []);

  return { ...state, sendText, pause, resume, reset, reconnect: connect };
}
