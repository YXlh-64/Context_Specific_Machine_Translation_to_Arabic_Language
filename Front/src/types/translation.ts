export interface TranslationVariant {
  id: string;
  text: string;
  rank: number;
  isEdited: boolean;
  isCustom?: boolean;
}

export interface TranslationSession {
  id: string;
  sourceText: string;
  sourceLanguage: 'en' | 'fr' | 'ar';
  detectedLanguage?: 'en' | 'fr';
  variants: TranslationVariant[];
  timestamp: Date;
  isFromFile?: boolean;
  fileName?: string;
}

export interface User {
  id: string;
  name: string;
}

export type AuthMode = 'login' | 'signup';
