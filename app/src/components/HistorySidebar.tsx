import { TranslationSession } from '@/types/translation';
import { Clock, ChevronLeft, ChevronRight, Trash2, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';
import { useI18n } from '@/contexts/I18nContext';

interface HistorySidebarProps {
  isCollapsed: boolean;
  onToggle: () => void;
  sessions: TranslationSession[];
  currentSessionId: string | null;
  onSelectSession: (session: TranslationSession) => void;
  onDeleteSession: (sessionId: string) => void;
}

export const HistorySidebar = ({
  isCollapsed,
  onToggle,
  sessions,
  currentSessionId,
  onSelectSession,
  onDeleteSession,
}: HistorySidebarProps) => {
  const { t, isRtl } = useI18n();

  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div
      className={cn(
        'h-full bg-sidebar border-sidebar-border flex flex-col transition-all duration-300 ease-out',
        isCollapsed ? 'w-12' : 'w-64',
        isRtl ? 'border-l' : 'border-r'
      )}
    >
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-3 border-b border-sidebar-border shrink-0">
        {!isCollapsed && (
          <div className="flex items-center gap-2 animate-fade-in">
            <Clock className="w-4 h-4 text-sidebar-foreground/70" />
            <span className="text-sm font-medium text-sidebar-foreground">{t.history}</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="h-8 w-8 hover:bg-sidebar-accent"
        >
          {isCollapsed ? (
            isRtl ? <ChevronLeft className="w-4 h-4 text-sidebar-foreground/70" /> : <ChevronRight className="w-4 h-4 text-sidebar-foreground/70" />
          ) : (
            isRtl ? <ChevronRight className="w-4 h-4 text-sidebar-foreground/70" /> : <ChevronLeft className="w-4 h-4 text-sidebar-foreground/70" />
          )}
        </Button>
      </div>

      {/* Sessions List */}
      {!isCollapsed && (
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {sessions.length === 0 ? (
              <div className="p-4 text-center">
                <p className="text-xs text-muted-foreground">
                  {t.noHistory}
                </p>
              </div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.id}
                  className={cn(
                    'group relative rounded-md cursor-pointer transition-all duration-150',
                    currentSessionId === session.id
                      ? 'bg-sidebar-accent'
                      : 'hover:bg-sidebar-accent/50'
                  )}
                >
                  <button
                    onClick={() => onSelectSession(session)}
                    className="w-full p-3 text-left"
                  >
                    <div className="flex items-center gap-2">
                      {session.isFromFile && (
                        <FileText className="w-3.5 h-3.5 text-primary shrink-0" />
                      )}
                      <p className="text-sm font-medium text-sidebar-foreground truncate flex-1">
                        {session.isFromFile && session.fileName
                          ? session.fileName
                          : truncateText(session.sourceText, 30)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-muted-foreground uppercase">
                        {session.detectedLanguage || session.sourceLanguage}
                      </span>
                      <span className="text-xs text-muted-foreground">→</span>
                      <span className="text-xs text-muted-foreground uppercase">AR</span>
                      <span className="text-xs text-muted-foreground ms-auto">
                        {format(new Date(session.timestamp), 'HH:mm')}
                      </span>
                    </div>
                  </button>
                  
                  {/* Delete button */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className={cn(
                      "absolute top-1/2 -translate-y-1/2 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/10 hover:text-destructive",
                      isRtl ? "left-1" : "right-1"
                    )}
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      )}

      {/* Collapsed state - just show icon */}
      {isCollapsed && sessions.length > 0 && (
        <div className="flex-1 flex items-start justify-center pt-4">
          <div className="w-6 h-6 rounded-full bg-sidebar-accent flex items-center justify-center">
            <span className="text-xs font-medium text-sidebar-foreground">
              {sessions.length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
