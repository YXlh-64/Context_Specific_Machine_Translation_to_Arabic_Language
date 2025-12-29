import { useState } from 'react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { TranslationVariant } from '@/types/translation';
import { GripVertical, Pencil, Check, X, Copy, CheckCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TranslationCardProps {
  variant: TranslationVariant;
  onEdit: (id: string, newText: string) => void;
  startInEditMode?: boolean;
}

export const TranslationCard = ({ variant, onEdit, startInEditMode = false }: TranslationCardProps) => {
  const [isEditing, setIsEditing] = useState(startInEditMode);
  const [editText, setEditText] = useState(variant.text);
  const [isCopied, setIsCopied] = useState(false);

  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: variant.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const handleSave = () => {
    onEdit(variant.id, editText);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditText(variant.text);
    setIsEditing(false);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(variant.text);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <div ref={setNodeRef} style={style}>
      <Card
        className={cn(
          'card-interactive border border-border bg-card',
          isDragging && 'opacity-50 shadow-elevated',
          variant.isEdited && 'ring-1 ring-primary/20'
        )}
      >
        <CardContent className="p-0">
          {/* Card Header */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-border/50">
            <div className="flex items-center gap-2">
              <button
                {...attributes}
                {...listeners}
                className="cursor-grab active:cursor-grabbing p-1 -ml-1 rounded hover:bg-muted transition-colors"
              >
                <GripVertical className="w-4 h-4 text-muted-foreground" />
              </button>
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Variation {variant.rank}
              </span>
              {variant.isCustom && (
                <span className="text-xs text-amber-600 bg-amber-500/10 px-2 py-0.5 rounded font-medium">
                  User Custom
                </span>
              )}
              {variant.isEdited && !variant.isCustom && (
                <span className="text-xs text-primary bg-primary/10 px-2 py-0.5 rounded">
                  Edited
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCopy}
                className="h-7 w-7 hover:bg-muted"
                disabled={isEditing}
              >
                {isCopied ? (
                  <CheckCheck className="w-3.5 h-3.5 text-primary" />
                ) : (
                  <Copy className="w-3.5 h-3.5 text-muted-foreground" />
                )}
              </Button>
              {!isEditing && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsEditing(true)}
                  className="h-7 w-7 hover:bg-muted"
                >
                  <Pencil className="w-3.5 h-3.5 text-muted-foreground" />
                </Button>
              )}
            </div>
          </div>

          {/* Card Body */}
          <div className="p-4">
            {isEditing ? (
              <div className="space-y-3">
                <Textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  className="min-h-[100px] rtl-textarea text-lg leading-relaxed resize-none bg-muted/30 border-border focus-ring"
                  dir="rtl"
                  autoFocus
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCancel}
                    className="h-8"
                  >
                    <X className="w-3.5 h-3.5 mr-1" />
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSave}
                    className="h-8"
                  >
                    <Check className="w-3.5 h-3.5 mr-1" />
                    Save
                  </Button>
                </div>
              </div>
            ) : (
              <p className="rtl-textarea text-lg leading-relaxed text-foreground min-h-[60px]">
                {variant.text}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
