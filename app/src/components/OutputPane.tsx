import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { TranslationVariant } from '@/types/translation';
import { TranslationCard } from './TranslationCard';
import { SkeletonCard } from './SkeletonCard';
import { Button } from '@/components/ui/button';
import { FileText, Plus, Send } from 'lucide-react';
import { useI18n } from '@/contexts/I18nContext';

interface OutputPaneProps {
  variants: TranslationVariant[];
  onReorder: (variants: TranslationVariant[]) => void;
  onEdit: (id: string, newText: string) => void;
  onAddCustom: () => void;
  onSubmitPreferences: () => void;
  isLoading: boolean;
  isSubmitting: boolean;
  newCustomId: string | null;
  targetLanguage: string;
  isRtl: boolean;
}

export const OutputPane = ({
  variants,
  onReorder,
  onEdit,
  onAddCustom,
  onSubmitPreferences,
  isLoading,
  isSubmitting,
  newCustomId,
  targetLanguage,
  isRtl,
}: OutputPaneProps) => {
  const { t } = useI18n();
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = variants.findIndex((v) => v.id === active.id);
      const newIndex = variants.findIndex((v) => v.id === over.id);

      const newVariants = arrayMove(variants, oldIndex, newIndex).map(
        (v, index) => ({
          ...v,
          rank: index + 1,
        })
      );

      onReorder(newVariants);
    }
  };

  const hasTranslations = variants.length > 0;

  return (
    <div className="h-full flex flex-col bg-surface-sunken">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-border shrink-0 bg-background">
        <div className="flex items-center gap-3">
          <FileText className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            {targetLanguage} Translations
          </span>
          {isRtl && (
            <span className="text-xs font-medium uppercase px-2 py-1 rounded bg-primary/10 text-primary">
              RTL
            </span>
          )}
        </div>
        {hasTranslations && !isLoading && (
          <span className="text-xs text-muted-foreground">
            Drag to reorder by preference
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 p-4 panel-scroll">
        {isLoading ? (
          <div className="space-y-4 animate-fade-in">
            <SkeletonCard rank={1} />
            <SkeletonCard rank={2} />
            <SkeletonCard rank={3} />
          </div>
        ) : hasTranslations ? (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={variants.map((v) => v.id)}
              strategy={verticalListSortingStrategy}
            >
              <div className="space-y-4">
                {variants.map((variant) => (
                  <TranslationCard
                    key={variant.id}
                    variant={variant}
                    onEdit={onEdit}
                    startInEditMode={variant.id === newCustomId}
                  />
                ))}
                
                {/* Add Custom Translation Button */}
                <Button
                  variant="outline"
                  onClick={onAddCustom}
                  className="w-full h-14 border-2 border-dashed border-border hover:border-primary/50 hover:bg-primary/5 transition-all"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  {t.draftCustom}
                </Button>

                {/* Submit Preferences Button */}
                <Button
                  onClick={onSubmitPreferences}
                  disabled={isSubmitting}
                  className="w-full h-12 mt-4 bg-primary hover:bg-primary/90 text-primary-foreground font-medium shadow-md transition-all"
                >
                  <Send className="w-4 h-4 mr-2" />
                  {isSubmitting ? t.translating : t.submitPreferences}
                </Button>
              </div>
            </SortableContext>
          </DndContext>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-4">
              <FileText className="w-8 h-8 text-muted-foreground/50" />
            </div>
            <p className="text-sm text-muted-foreground mb-1">
              No translations yet
            </p>
            <p className="text-xs text-muted-foreground/70">
              Enter text and click "Translate" to get started
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
