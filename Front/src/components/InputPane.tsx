import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Sparkles, Loader2 } from 'lucide-react';
import { FileUpload, UploadedFile } from './FileUpload';
import { useI18n } from '@/contexts/I18nContext';

interface InputPaneProps {
  text: string;
  onTextChange: (text: string) => void;
  onTranslate: () => void;
  isTranslating: boolean;
  direction: 'ltr' | 'rtl';
  placeholder: string;
  targetLanguage: string;
  uploadedFile: UploadedFile | null;
  onFileUpload: (file: UploadedFile | null) => void;
}

export const InputPane = ({
  text,
  onTextChange,
  onTranslate,
  isTranslating,
  direction,
  placeholder,
  targetLanguage,
  uploadedFile,
  onFileUpload,
}: InputPaneProps) => {
  const { t } = useI18n();
  const canTranslate = (text.trim().length > 0 || uploadedFile) && !isTranslating;

  return (
    <div className="h-full flex flex-col bg-background">
      {/* File Upload Section */}
      <div className="p-4 border-b border-border shrink-0">
        <FileUpload
          onFileUpload={onFileUpload}
          uploadedFile={uploadedFile}
          disabled={isTranslating}
        />
      </div>

      {/* Text Area */}
      <div className="flex-1 p-4 overflow-hidden">
        <Textarea
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder={uploadedFile ? '' : placeholder}
          dir={direction}
          className={`h-full w-full resize-none text-base leading-relaxed bg-transparent border-0 focus:ring-0 focus-visible:ring-0 focus-visible:ring-offset-0 p-0 placeholder:text-muted-foreground/50 panel-scroll ${direction === 'rtl' ? 'text-right' : 'text-left'}`}
          disabled={isTranslating || !!uploadedFile}
        />
      </div>

      {/* Footer with Translate Button */}
      <div className="h-16 flex items-center justify-between px-4 border-t border-border shrink-0">
        <div className="text-xs text-muted-foreground">
          {text.length > 0 && `${text.split(/\s+/).filter(Boolean).length} ${t.words}`}
        </div>
        <Button
          onClick={onTranslate}
          disabled={!canTranslate}
          className="h-9 px-4 gap-2"
        >
          {isTranslating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {t.translating}
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              {t.translateTo} {targetLanguage}
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
