import { useState, useCallback, DragEvent } from 'react';
import { Paperclip, X, FileText, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useI18n } from '@/contexts/I18nContext';
import { cn } from '@/lib/utils';

export interface UploadedFile {
  name: string;
  type: string;
  size: number;
  extractedText: string;
}

interface FileUploadProps {
  onFileUpload: (file: UploadedFile | null) => void;
  uploadedFile: UploadedFile | null;
  disabled?: boolean;
}

// Upload file to backend for real text extraction
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5002/api';

const uploadAndExtractText = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append('file', file);

  const resp = await fetch(`${API_BASE}/upload-file`, {
    method: 'POST',
    body: formData,
    // Ensure we allow cross-origin requests if running on a different origin
    mode: 'cors',
  });

  if (!resp.ok) {
    let msg = 'Upload failed';
    try {
      const err = await resp.json();
      const detail = err.detail ? `: ${err.detail}` : '';
      msg = (err.error || msg) + detail;
    } catch (e) {
      msg = `${resp.status} ${resp.statusText}`;
    }
    throw new Error(msg);
  }

  const data = await resp.json();
  return data.text || '';
};

export const FileUpload = ({ onFileUpload, uploadedFile, disabled }: FileUploadProps) => {
  const { t } = useI18n();
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const acceptedTypes = ['.pdf', '.docx', '.txt'];
  const acceptedMimeTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
  ];

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFile = async (file: File) => {
    const isAccepted = acceptedMimeTypes.includes(file.type) || acceptedTypes.some(ext => file.name.toLowerCase().endsWith(ext.toLowerCase()));
    if (!isAccepted) {
      const msg = 'Unsupported file type. Please upload PDF, DOCX, or TXT files.';
      try {
        const { toast } = await import('@/hooks/use-toast');
        toast({ title: 'Invalid file', description: msg });
      } catch (_) {
        alert(msg);
      }
      return;
    }

    setIsProcessing(true);
    try {
      const extractedText = await uploadAndExtractText(file);
      onFileUpload({
        name: file.name,
        type: file.type,
        size: file.size,
        extractedText,
      });
    } catch (e: any) {
      console.error('File extraction failed', e);
      const msg = e?.message || 'Failed to extract text from file';
      try {
        const { toast } = await import('@/hooks/use-toast');
        toast({ title: 'Upload failed', description: msg });
      } catch (_) {
        alert(msg);
      }
      onFileUpload(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = useCallback(async (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (disabled) return;

    const file = e.dataTransfer.files[0];
    if (file) {
      await processFile(file);
    }
  }, [disabled, onFileUpload]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await processFile(file);
    }
    e.target.value = '';
  };

  const handleRemove = () => {
    onFileUpload(null);
  };

  if (uploadedFile) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg border border-border">
        <FileText className="w-4 h-4 text-primary shrink-0" />
        <span className="text-sm text-foreground truncate flex-1">
          {uploadedFile.name}
        </span>
        <span className="text-xs text-muted-foreground shrink-0">
          {(uploadedFile.size / 1024).toFixed(1)} KB
        </span>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleRemove}
          className="h-6 w-6 shrink-0 hover:bg-destructive/10 hover:text-destructive"
          title={t.removeFile}
          disabled={disabled}
        >
          <X className="w-3.5 h-3.5" />
        </Button>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        'relative border-2 border-dashed rounded-lg transition-all',
        isDragging 
          ? 'border-primary bg-primary/5' 
          : 'border-border/50 hover:border-border',
        disabled && 'opacity-50 pointer-events-none'
      )}
    >
      <label className="flex flex-col items-center justify-center gap-2 p-4 cursor-pointer">
        <input
          type="file"
          accept={acceptedTypes.join(',')}
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled || isProcessing}
        />
        {isProcessing ? (
          <>
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-muted-foreground">Processing...</span>
          </>
        ) : (
          <>
            <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">
              {isDragging ? (
                <Upload className="w-5 h-5 text-primary" />
              ) : (
                <Paperclip className="w-5 h-5 text-muted-foreground" />
              )}
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                {isDragging ? t.dropFileHere : t.uploadFile}
              </p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                {t.supportedFormats}
              </p>
            </div>
          </>
        )}
      </label>
    </div>
  );
};
