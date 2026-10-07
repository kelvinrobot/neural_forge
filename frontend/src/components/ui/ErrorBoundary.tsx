import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "./Button";
import { Icon } from "./Icon";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Neural Forge UI error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto flex max-w-lg flex-col items-center gap-4 py-20 text-center" role="alert">
        <span className="rounded-full bg-amber-50 p-4 text-amber-600 dark:bg-amber-950">
          <Icon name="alert" size={32} />
        </span>
        <h1 className="text-xl font-semibold">Something went wrong on this screen</h1>
        <p className="text-sm text-ink-600 dark:text-ink-300">
          The page hit an unexpected problem. Your camera has been released. You can return to the start and begin the examination again.
        </p>
        <p className="max-w-full break-words rounded-lg bg-ink-100 px-3 py-2 font-mono text-xs text-ink-600 dark:bg-ink-800 dark:text-ink-300">
          {this.state.error.message}
        </p>
        <Button
          onClick={() => {
            window.location.assign("/");
          }}
        >
          Back to home
        </Button>
      </div>
    );
  }
}
