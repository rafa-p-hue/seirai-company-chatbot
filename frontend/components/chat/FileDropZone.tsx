"use client";

import { useId, useRef, useState, type DragEvent } from "react";

import { UploadCloudIcon } from "@/components/chat/icons";
import {
  SUPPORTED_FORMAT_LABEL,
  UPLOAD_ACCEPT_EXTENSIONS,
} from "@/lib/supported-formats";

type FileDropZoneProps = {
  onFiles: (files: FileList | File[]) => void;
  disabled?: boolean;
  compact?: boolean;
};

function openPicker(input: HTMLInputElement | null) {
  if (!input || input.disabled) return;
  // Reset so selecting the same file again still fires onChange.
  input.value = "";
  input.click();
}

export function FileDropZone({
  onFiles,
  disabled = false,
  compact = false,
}: FileDropZoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function emitFiles(files: FileList | null) {
    if (!files || files.length === 0 || disabled) return;
    onFiles(files);
  }

  function onDragEnter(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) setDragging(true);
  }

  function onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) setDragging(true);
  }

  function onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    const next = event.relatedTarget as Node | null;
    if (!event.currentTarget.contains(next)) setDragging(false);
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
    emitFiles(event.dataTransfer.files);
  }

  if (compact) {
    return (
      <div className="relative">
        <button
          type="button"
          disabled={disabled}
          onClick={() => openPicker(inputRef.current)}
          className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 transition hover:border-slate-300 hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-400 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Attach files to this chat"
          title={`Attach files (${SUPPORTED_FORMAT_LABEL})`}
        >
          <UploadCloudIcon className="h-4 w-4" />
        </button>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={UPLOAD_ACCEPT_EXTENSIONS}
          multiple
          disabled={disabled}
          className="pointer-events-none absolute h-px w-px opacity-0"
          tabIndex={-1}
          onChange={(event) => {
            emitFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
    );
  }

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-dashed transition ${
        dragging
          ? "border-sky-500 bg-sky-50/80 shadow-sm"
          : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/60"
      } ${disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {/* Native file input covers the zone so clicks always open the picker. */}
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={UPLOAD_ACCEPT_EXTENSIONS}
        multiple
        disabled={disabled}
        aria-label="Upload files to this chat"
        className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
        onChange={(event) => {
          emitFiles(event.target.files);
          event.target.value = "";
        }}
      />

      <div className="pointer-events-none px-6 py-7 text-center">
        <UploadCloudIcon
          className={`mx-auto h-11 w-11 ${dragging ? "text-sky-600" : "text-slate-400"}`}
        />
        <p className="mt-3 text-sm font-medium text-slate-800">
          Drop your files here, or{" "}
          <span className="font-semibold text-sky-800 underline-offset-2">
            browse
          </span>
        </p>
        <p className="mt-1.5 text-xs text-slate-500">
          {SUPPORTED_FORMAT_LABEL} · max 10 MB each · attached to this chat
        </p>
      </div>
    </div>
  );
}
