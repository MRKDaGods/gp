"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { ImagePlus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ReIDUploadImage {
  id: string;
  name: string;
  imageBase64: string;
  previewUrl: string;
}

interface ImageUploaderProps {
  label: string;
  images: ReIDUploadImage[];
  onChange: (images: ReIDUploadImage[]) => void;
  maxFiles?: number;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export function ImageUploader({ label, images, onChange, maxFiles = 50 }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const remaining = useMemo(() => Math.max(0, maxFiles - images.length), [images.length, maxFiles]);

  const addFiles = useCallback(async (fileList: FileList | File[]) => {
    const files = Array.from(fileList).filter((file) => file.type.startsWith("image/")).slice(0, remaining);
    const loaded = await Promise.all(files.map(async (file, index) => ({
      id: `${label.toLowerCase().replace(/\s+/g, "-")}-${Date.now()}-${index}`,
      name: file.name,
      imageBase64: await readFileAsDataUrl(file),
      previewUrl: URL.createObjectURL(file),
    })));
    onChange([...images, ...loaded]);
  }, [images, label, onChange, remaining]);

  const removeImage = useCallback((id: string) => {
    const image = images.find((item) => item.id === id);
    if (image) URL.revokeObjectURL(image.previewUrl);
    onChange(images.filter((item) => item.id !== id));
  }, [images, onChange]);

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold">{label}</h2>
        <span className="text-xs text-muted-foreground">{images.length}/{maxFiles}</span>
      </div>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          void addFiles(event.dataTransfer.files);
        }}
        className={cn(
          "flex min-h-28 w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed p-4 text-sm text-muted-foreground transition-colors",
          isDragging ? "border-primary bg-primary/10 text-foreground" : "hover:border-primary/70 hover:bg-muted/40"
        )}
      >
        <ImagePlus className="h-5 w-5" />
        <span>{remaining > 0 ? "Drop images or browse" : "Image limit reached"}</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(event) => {
          if (event.target.files) void addFiles(event.target.files);
          event.currentTarget.value = "";
        }}
      />
      {images.length > 0 && (
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6">
          {images.map((image) => (
            <div key={image.id} className="group relative aspect-square overflow-hidden rounded-md border bg-muted">
              <img src={image.previewUrl} alt={image.name} className="h-full w-full object-cover" />
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="absolute right-1 top-1 h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100"
                onClick={() => removeImage(image.id)}
                aria-label={`Remove ${image.name}`}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}