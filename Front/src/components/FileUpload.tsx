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

// Mock text extraction - simulates extracting text from different file types
const mockExtractText = async (file: File): Promise<string> => {
  await new Promise(resolve => setTimeout(resolve, 500));
  
  const mockTexts: Record<string, string> = {
    pdf: `This is extracted text from the PDF document "${file.name}". It contains professional content that needs to be translated. The document discusses various topics including business communications, technical specifications, and general correspondence.`,
    docx: `This is extracted text from the Word document "${file.name}". It includes formatted paragraphs, headings, and body text that require translation services. The content covers meeting notes, project updates, and action items.`,
    txt: `This is the content of the text file "${file.name}". Plain text files are straightforward to process. This sample includes multiple sentences that demonstrate the translation workflow.`,
  };

  const ext = file.name.split('.').pop()?.toLowerCase() || 'txt';
  return mockTexts[ext] || mockTexts.txt;
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
    if (!acceptedMimeTypes.includes(file.type) && !acceptedTypes.some(ext => file.name.endsWith(ext))) {
      return;
    }

    setIsProcessing(true);
    try {
      const extractedText = await mockExtractText(file);
      onFileUpload({
        name: file.name,
        type: file.type,
        size: file.size,
        extractedText,
      });
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
