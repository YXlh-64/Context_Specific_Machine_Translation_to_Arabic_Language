import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Button } from '@/components/ui/button';
import { ArrowLeftRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export type SourceLanguage = 'en' | 'fr' | 'ar';
export type TargetLanguage = 'ar' | 'en' | 'fr';

interface LanguageBarProps {
  sourceLanguage: SourceLanguage;
  targetLanguage: TargetLanguage;
  onSourceChange: (lang: SourceLanguage) => void;
  onTargetChange: (lang: TargetLanguage) => void;
  onSwap: () => void;
  isSwapped: boolean;
}

export const LanguageBar = ({
  sourceLanguage,
  targetLanguage,
  onSourceChange,
  onTargetChange,
  onSwap,
  isSwapped,
}: LanguageBarProps) => {
  const sourceOptions = isSwapped
    ? [{ value: 'ar', label: 'Arabic' }]
    : [
        { value: 'en', label: 'English' },
        { value: 'fr', label: 'French' },
      ];

  const targetOptions = isSwapped
    ? [
        { value: 'en', label: 'English' },
        { value: 'fr', label: 'French' },
      ]
    : [{ value: 'ar', label: 'Arabic' }];

  return (
    <div className="h-12 flex items-center justify-between px-4 border-b border-border bg-surface-sunken shrink-0">
      {/* Source Language Selection */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-2">
          From
        </span>
        <ToggleGroup
          type="single"
          value={sourceLanguage}
          onValueChange={(value) => value && onSourceChange(value as SourceLanguage)}
          className="gap-1"
        >
          {sourceOptions.map((option) => (
            <ToggleGroupItem
              key={option.value}
              value={option.value}
              className={cn(
                'h-8 px-3 text-xs font-medium rounded-md transition-all',
                'data-[state=on]:bg-primary data-[state=on]:text-primary-foreground',
                'data-[state=off]:bg-muted/50 data-[state=off]:text-muted-foreground',
                'hover:bg-muted'
              )}
            >
              {option.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Swap Button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={onSwap}
        className={cn(
          'h-8 w-8 rounded-full transition-all hover:bg-primary/10',
          isSwapped && 'rotate-180'
        )}
        title="Swap languages"
      >
        <ArrowLeftRight className="w-4 h-4 text-primary" />
      </Button>

      {/* Target Language Selection */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-2">
          To
        </span>
        <ToggleGroup
          type="single"
          value={targetLanguage}
          onValueChange={(value) => value && onTargetChange(value as TargetLanguage)}
          className="gap-1"
        >
          {targetOptions.map((option) => (
            <ToggleGroupItem
              key={option.value}
              value={option.value}
              className={cn(
                'h-8 px-3 text-xs font-medium rounded-md transition-all',
                'data-[state=on]:bg-primary data-[state=on]:text-primary-foreground',
                'data-[state=off]:bg-muted/50 data-[state=off]:text-muted-foreground',
                'hover:bg-muted'
              )}
            >
              {option.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
    </div>
  );
};
