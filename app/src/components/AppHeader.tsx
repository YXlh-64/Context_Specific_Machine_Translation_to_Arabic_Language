import { Button } from '@/components/ui/button';
import { User } from '@/types/translation';
import { Languages, LogOut, User as UserIcon } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { LanguageSelector } from './LanguageSelector';
import { useI18n } from '@/contexts/I18nContext';

interface AppHeaderProps {
  user: User | null;
  onLogin: () => void;
  onLogout: () => void;
}

export const AppHeader = ({ user, onLogin, onLogout }: AppHeaderProps) => {
  const { t } = useI18n();

  return (
    <header className="h-14 border-b border-border bg-background flex items-center justify-between px-4 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <Languages className="w-4 h-4 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-base font-semibold text-foreground leading-none">
            {t.appName}
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            {t.appSubtitle}
          </p>
        </div>
      </div>

      {/* Right Side: Language Selector + User Menu */}
      <div className="flex items-center gap-2">
        <LanguageSelector />
        
        {user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-9 gap-2 px-3">
                <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center">
                  <UserIcon className="w-3.5 h-3.5 text-primary" />
                </div>
                <span className="text-sm font-medium">{user.name}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48 bg-popover border border-border z-50">
              <div className="px-2 py-1.5">
                <p className="text-sm font-medium text-foreground">{user.name}</p>
                <p className="text-xs text-muted-foreground">{t.loggedIn}</p>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onLogout} className="text-destructive focus:text-destructive cursor-pointer">
                <LogOut className="w-4 h-4 mr-2" />
                {t.signOut}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button variant="outline" size="sm" onClick={onLogin} className="h-9">
            <UserIcon className="w-4 h-4 mr-2" />
            {t.signIn}
          </Button>
        )}
      </div>
    </header>
  );
};
