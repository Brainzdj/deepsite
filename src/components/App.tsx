/* eslint-disable @typescript-eslint/no-explicit-any */
import { useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import classNames from "classnames";
import { editor } from "monaco-editor";
import {
  useMount,
  useUnmount,
  useEvent,
  createBreakpoint,
  useLocalStorage,
} from "react-use";
import { toast } from "react-toastify";

import Header from "./header/header";
import DeployButton from "./deploy-button/deploy-button";
import { defaultHTML } from "../utils/consts";
import Tabs from "./tabs/tabs";
import AskAI from "./ask-ai/ask-ai";
import { Auth } from "../utils/types";
import Preview from "./preview/preview";

const useBreakpoint = createBreakpoint({
  md: 768,
  lg: 1024,
});

function App() {
  const breakpoint = useBreakpoint();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [htmlStorage, _, removeHtmlStorage] = useLocalStorage("html_content");

  const preview = useRef<HTMLDivElement>(null);
  const editor = useRef<HTMLDivElement>(null);
  const resizer = useRef<HTMLDivElement>(null);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const [isResizing, setIsResizing] = useState(false);
  const [error, setError] = useState(false);
  const [html, setHtml] = useState((htmlStorage as string) ?? defaultHTML);
  const [isAiWorking, setisAiWorking] = useState(false);
  const [auth, setAuth] = useState<Auth | undefined>(undefined);

  const fetchMe = async () => {
    const res = await fetch("/api/@me");
    if (res.ok) {
      const data = await res.json();
      setAuth(data);
    } else {
      setAuth(undefined);
    }
  };

  const handleResize = (e: MouseEvent) => {
    if (!editor.current || !preview.current || !resizer.current) return;
    const editorWidth = e.clientX;
    const previewWidth = window.innerWidth - editorWidth - 4;
    editor.current.style.width = `${editorWidth}px`;
    preview.current.style.width = `${previewWidth}px`;
  };

  const handleMouseDown = () => {
    setIsResizing(true);
    document.addEventListener("mousemove", handleResize);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseUp = () => {
    setIsResizing(false);
    document.removeEventListener("mousemove", handleResize);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  useEvent("beforeunload", (e) => {
    if (isAiWorking || html !== defaultHTML) {
      e.preventDefault();
      return "";
    }
  });

  useMount(() => {
    fetchMe();
    if (htmlStorage) {
      removeHtmlStorage();
      toast.warn("Previous HTML content restored from local storage.");
    }
    if (!editor.current || !preview.current) return;

    if (breakpoint === "lg") {
      // Set initial sizes
      const initialEditorWidth = window.innerWidth / 2;
      const initialPreviewWidth = window.innerWidth - initialEditorWidth - 4;
      editor.current.style.width = `${initialEditorWidth}px`;
      preview.current.style.width = `${initialPreviewWidth}px`;
    }

    if (!resizer.current) return;
    resizer.current.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("resize", () => handleMouseDown);
  });

  useUnmount(() => {
    document.removeEventListener("mousemove", handleResize);
    document.removeEventListener("mouseup", handleMouseUp);
    if (resizer.current) {
      resizer.current.removeEventListener("mousedown", handleMouseDown);
    }
    window.removeEventListener("resize", () => handleMouseDown);
  });

  return (
    <div className="h-screen bg-gray-950 font-sans overflow-hidden">
      <Header>
        <DeployButton html={html} error={error} auth={auth} />
      </Header>
      <main className="max-lg:flex-col flex w-full">
        <div
          ref={editor}
          className="w-full h-[30dvh] lg:h-full relative"
          onClick={(e) => {
            if (isAiWorking) {
              e.preventDefault();
              e.stopPropagation();
              toast.warn("Please wait for the AI to finish working.");
            }
          }}
        >
          <Tabs />
          <Editor
            language="html"
            theme="vs-dark"
            className={classNames(
              "h-[calc(30dvh-41px)] lg:h-[calc(100dvh-96px)]",
              {
                "pointer-events-none": isAiWorking,
              }
            )}
            value={html}
            onValidate={(markers) => {
              if (markers?.length > 0) {
                setError(true);
              }
            }}
            onChange={(value) => {
              const newValue = value ?? "";
              setHtml(newValue);
              setError(false);
            }}
            onMount={(editor) => (editorRef.current = editor)}
          />
          <AskAI
            html={html}
            setHtml={setHtml}
            isAiWorking={isAiWorking}
            setisAiWorking={setisAiWorking}
            onScrollToBottom={() => {
              editorRef.current?.revealLine(
                editorRef.current?.getModel()?.getLineCount() ?? 0
              );
            }}
          />
        </div>
        <div
          ref={resizer}
          className="bg-gray-700 hover:bg-blue-500 w-2 cursor-col-resize h-[calc(100dvh-54px)] max-lg:hidden"
        />
        <Preview
          html={html}
          isResizing={isResizing}
          isAiWorking={isAiWorking}
          ref={preview}
        />
      </main>
    </div>
  );
}

export default App;
