import { useState, useCallback, useEffect } from 'react';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import { AppHeader } from '@/components/AppHeader';
import { AuthModal } from '@/components/AuthModal';
import { HistorySidebar } from '@/components/HistorySidebar';
import { InputPane } from '@/components/InputPane';
import { OutputPane } from '@/components/OutputPane';
import { LanguageBar, SourceLanguage, TargetLanguage } from '@/components/LanguageBar';
import { User, TranslationSession, TranslationVariant } from '@/types/translation';
// import { mockTranslate } from '@/lib/mockTranslation'; // We will replace this with a real API call
import { toast } from '@/hooks/use-toast';
import { useI18n } from '@/contexts/I18nContext';
import { UploadedFile } from '@/components/FileUpload';
import { cn } from '@/lib/utils';

const LANGUAGE_STORAGE_KEY = 'linguist-bridge-language-prefs';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

interface LanguagePrefs {
  sourceLanguage: SourceLanguage;
  targetLanguage: TargetLanguage;
  isSwapped: boolean;
}

const getStoredLanguagePrefs = (): LanguagePrefs => {
  try {
    const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error('Failed to load language preferences', e);
  }
  return { sourceLanguage: 'en', targetLanguage: 'ar', isSwapped: false };
};

const saveLanguagePrefs = (prefs: LanguagePrefs) => {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, JSON.stringify(prefs));
  } catch (e) {
    console.error('Failed to save language preferences', e);
  }
};

const Index = () => {
  const { t, isRtl } = useI18n();

  // Auth state
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  // UI state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Language state (with localStorage persistence)
  const storedPrefs = getStoredLanguagePrefs();
  const [sourceLanguage, setSourceLanguage] = useState<SourceLanguage>(storedPrefs.sourceLanguage as SourceLanguage);
  const [targetLanguage, setTargetLanguage] = useState<TargetLanguage>(storedPrefs.targetLanguage);
  const [isSwapped, setIsSwapped] = useState(storedPrefs.isSwapped);

  // Translation state
  const [inputText, setInputText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [isSubmittingPreferences, setIsSubmittingPreferences] = useState(false);
  const [variants, setVariants] = useState<TranslationVariant[]>([]);
  const [newCustomId, setNewCustomId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);

  // History state
  const [sessions, setSessions] = useState<TranslationSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  // Persist language preferences
  useEffect(() => {
    saveLanguagePrefs({ sourceLanguage, targetLanguage, isSwapped });
  }, [sourceLanguage, targetLanguage, isSwapped]);


  // Derived values
  const inputDirection = isSwapped ? 'rtl' : 'ltr';
  const inputPlaceholder = isSwapped
    ? 'اكتب للترجمة...'
    : sourceLanguage === 'fr'
    ? t.typeToTranslate.replace('...', '...')
    : t.typeToTranslate;

  const targetLanguageLabel = isSwapped
    ? targetLanguage === 'en'
      ? t.english
      : t.french
    : t.arabic;

  // Handle file upload
  const handleFileUpload = (file: UploadedFile | null) => {
    setUploadedFile(file);
    if (file) {
      setInputText(file.extractedText);
      toast({
        title: t.fileUploaded,
        description: t.textExtracted,
      });
    } else {
      setInputText('');
    }
  };

  const handleSourceChange = (lang: SourceLanguage) => {
    setSourceLanguage(lang);
  };

  const handleTargetChange = (lang: TargetLanguage) => {
    setTargetLanguage(lang);
  };

  const handleSwap = () => {
    if (isSwapped) {
      // Going back to normal mode
      setIsSwapped(false);
      setSourceLanguage('en');
      setTargetLanguage('ar');
    } else {
      // Swapping to Arabic → English/French
      setIsSwapped(true);
      setSourceLanguage('ar');
      setTargetLanguage('en');
    }
    // Clear current input when swapping
    setInputText('');
    setVariants([]);
    setUploadedFile(null);
  };

  const handleAuth = (authenticatedUser: User) => {
    setUser(authenticatedUser);
    toast({
      title: t.welcome,
      description: `${t.signedInAs} ${authenticatedUser.name}`,
    });
  };

  const handleLogout = () => {
    setUser(null);
    toast({
      title: t.signedOut,
      description: t.seeYou,
    });
  };

  const handleTranslate = useCallback(async () => {
    if (!inputText.trim() && !uploadedFile) return;

    setIsTranslating(true);
    setVariants([]);

    try {
      const effectiveSourceLang = isSwapped ? 'ar' : sourceLanguage;
      const textToTranslate = uploadedFile ? uploadedFile.extractedText : inputText;

      const response = await fetch(`${API_URL}/translate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textToTranslate,
          source_language: effectiveSourceLang,
          target_language: isSwapped ? targetLanguage : 'ar',
          domain: 'general', // You can make this dynamic later
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // The backend returns a list of strings. We need to convert them to TranslationVariant objects.
      const backendVariants: TranslationVariant[] = data.translations.map((text: string, index: number) => ({
        id: `variant-${Date.now()}-${index}`,
        text,
        rank: index + 1,
      }));

      setVariants(backendVariants);

      // Create new session
      const newSession: TranslationSession = {
        id: `session-${Date.now()}`,
        sourceText: textToTranslate,
        sourceLanguage: effectiveSourceLang,
        variants: backendVariants,
        timestamp: new Date(),
        isFromFile: !!uploadedFile,
        fileName: uploadedFile?.name,
      };

      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);

      toast({
        title: t.translationComplete,
        description: `${backendVariants.length} ${targetLanguageLabel} ${t.variationsGenerated}`,
      });
    } catch (error) {
      toast({
        title: t.translationFailed,
        description: t.tryAgain,
        variant: 'destructive',
      });
    } finally {
      setIsTranslating(false);
    }
  }, [inputText, sourceLanguage, isSwapped, targetLanguageLabel, uploadedFile, t]);

  const handleReorder = (newVariants: TranslationVariant[]) => {
    setVariants(newVariants);

    // Update the current session with new order
    if (currentSessionId) {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId ? { ...s, variants: newVariants } : s
        )
      );
    }
  };

  const handleEdit = (variantId: string, newText: string) => {
    const updatedVariants = variants.map((v) =>
      v.id === variantId ? { ...v, text: newText, isEdited: true } : v
    );
    setVariants(updatedVariants);

    // Update the current session
    if (currentSessionId) {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId ? { ...s, variants: updatedVariants } : s
        )
      );
    }

    // Clear the newCustomId after first edit save
    if (newCustomId === variantId) {
      setNewCustomId(null);
    }

    toast({
      title: t.translationUpdated,
      description: t.editSaved,
    });
  };

  const handleAddCustomTranslation = useCallback(() => {
    const customId = `custom-${Date.now()}`;
    const newVariant: TranslationVariant = {
      id: customId,
      text: '',
      rank: variants.length + 1,
      isEdited: false,
      isCustom: true,
    };

    const updatedVariants = [...variants, newVariant];
    setVariants(updatedVariants);
    setNewCustomId(customId);

    // Update the current session
    if (currentSessionId) {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === currentSessionId ? { ...s, variants: updatedVariants } : s
        )
      );
    }
  }, [variants, currentSessionId]);

  const handleSubmitPreferences = useCallback(async () => {
    if (variants.length === 0) return;

    setIsSubmittingPreferences(true);
    
    try {
      // Prepare the preference data
      const preferenceData = {
        sessionId: currentSessionId,
        rankings: variants.map((v, index) => ({
          variantId: v.id,
          rank: index + 1,
          text: v.text,
          isCustom: v.isCustom || false,
          isEdited: v.isEdited || false,
        })),
        selectedVariantId: variants[0]?.id, // Top-ranked is selected
        timestamp: new Date().toISOString(),
      };

      // Log the data (for now, since backend is removed)
      console.log('Submitting preferences:', preferenceData);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));

      toast({
        title: t.preferencesSubmitted,
        description: t.preferencesSubmittedDesc,
      });

      // Clear everything after successful submission
      setVariants([]);
      setInputText('');
      setCurrentSessionId(null);
      setUploadedFile(null);
      setNewCustomId(null);
    } catch (error) {
      toast({
        title: t.translationFailed,
        description: t.tryAgain,
        variant: 'destructive',
      });
    } finally {
      setIsSubmittingPreferences(false);
    }
  }, [variants, currentSessionId, t]);

  const handleSelectSession = (session: TranslationSession) => {
    setInputText(session.sourceText);
    setVariants(session.variants);
    setCurrentSessionId(session.id);
    setNewCustomId(null);
  };

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));

    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
      setVariants([]);
      setInputText('');
    }

    toast({
      title: t.sessionDeleted,
      description: t.removedFromHistory,
    });
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background">
      {/* App Header */}
      <AppHeader
        user={user}
        onLogin={() => setShowAuthModal(true)}
        onLogout={handleLogout}
      />

      {/* Main Content */}
      <div className={cn("flex-1 flex overflow-hidden", isRtl && "flex-row-reverse")}>
        {/* History Sidebar */}
        <HistorySidebar
          isCollapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onDeleteSession={handleDeleteSession}
        />

        {/* Main Workspace */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Language Selection Bar */}
          <LanguageBar
            sourceLanguage={sourceLanguage}
            targetLanguage={targetLanguage}
            onSourceChange={handleSourceChange}
            onTargetChange={handleTargetChange}
            onSwap={handleSwap}
            isSwapped={isSwapped}
          />

          {/* Resizable Workspace */}
          <div className="flex-1 overflow-hidden">
            <ResizablePanelGroup direction="horizontal" className="h-full">
              {/* Input Panel */}
              <ResizablePanel defaultSize={50} minSize={30}>
                <InputPane
                  text={inputText}
                  onTextChange={setInputText}
                  onTranslate={handleTranslate}
                  isTranslating={isTranslating}
                  direction={inputDirection}
                  placeholder={inputPlaceholder}
                  targetLanguage={targetLanguageLabel}
                  uploadedFile={uploadedFile}
                  onFileUpload={handleFileUpload}
                />
              </ResizablePanel>

              {/* Resize Handle */}
              <ResizableHandle withHandle className="bg-border hover:bg-primary/20 transition-colors" />

              {/* Output Panel */}
              <ResizablePanel defaultSize={50} minSize={30}>
                <OutputPane
                  variants={variants}
                  onReorder={handleReorder}
                  onEdit={handleEdit}
                  onAddCustom={handleAddCustomTranslation}
                  onSubmitPreferences={handleSubmitPreferences}
                  isLoading={isTranslating}
                  isSubmitting={isSubmittingPreferences}
                  newCustomId={newCustomId}
                  targetLanguage={targetLanguageLabel}
                  isRtl={!isSwapped}
                />
              </ResizablePanel>
            </ResizablePanelGroup>
          </div>
        </div>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuth={handleAuth}
      />
    </div>
  );
};

export default Index;

