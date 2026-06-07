/**
 * SafeBoundary — 부분 렌더 실패가 페이지 전체를 빈 화면으로 만들지 않도록 격리.
 * children에서 throw가 발생하면 콘솔에 에러를 찍고 children 부분만 숨긴다.
 */
import { Component, type ReactNode } from "react";

interface Props { children: ReactNode; label?: string }
interface State { hasError: boolean }

export default class SafeBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error(`[SafeBoundary:${this.props.label ?? "unknown"}]`, error, info);
  }

  render() {
    if (this.state.hasError) return null;
    return this.props.children;
  }
}
