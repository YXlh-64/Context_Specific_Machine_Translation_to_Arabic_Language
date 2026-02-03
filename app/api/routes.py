
from flask import Blueprint, request, jsonify
import requests
from datetime import datetime
import logging

from rag_service import get_prompting_service

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

# Mock translation function - kept as fallback
def mock_translate(text, source_lang, target_lang, n=3):
    """Return a list of up to `n` mock translation variants.

    Used as fallback when RAG services are unavailable.
    """
    mock_translations_base = {
        'en-ar': [
            f"ترجمة تجريبية للنص: {text}",
            f"ترجمة بديلة: {text}",
            f"ترجمة مختصرة: {text}"
        ],
        'fr-ar': [
            f"ترجمة تجريبية من الفرنسية: {text}",
            f"ترجمة بديلة من الفرنسية: {text}",
            f"ترجمة مختصرة من الفرنسية: {text}"
        ],
        'ar-en': [
            f"Mock translation to English: {text}",
            f"Alternative English translation: {text}",
            f"Concise English translation: {text}"
        ],
        'ar-fr': [
            f"Traduction simulée vers le français: {text}",
            f"Traduction alternative vers le français: {text}",
            f"Traduction concise vers le français: {text}"
        ],
    }

    key = f"{source_lang}-{target_lang}"
    variants = mock_translations_base.get(key)
    if not variants:
        # Fallback: generate simple numbered variants
        variants = [f"Mock translation ({i+1}): {text}" for i in range(n)]

    # Ensure we only return up to n variants
    return variants[:n]


@api_bp.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Translation API Server',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'translate': '/api/translate',
            'detect_language': '/api/detect-language'
        }
    })

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'translation-api'
    })

@api_bp.route('/translate', methods=['POST'])
def translate():
    """
    Main translation endpoint - supports DeepL (free, default) and OpenRouter (paid)
    
    Parameters:
    - text: Text to translate
    - source_language: Source language code (en, fr, ar)
    - target_language: Target language code (en, fr, ar)
    - domain: Optional domain/context (default: 'general')
    - translation_service: 'deepl' (free, default) or 'openrouter' (paid, multiple variants)
    - num_variants: Number of translation variants (only for openrouter, default: 3)
    - auto_chunk: Auto-chunk long texts (default: True)
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text', '').strip()
        source_language = data.get('source_language', 'en')
        target_language = data.get('target_language', 'ar')
        domain = data.get('domain', 'general')  # Optional domain parameter
        translation_service_type = 'openrouter'  # Always use OpenRouter - DeepL disabled
        num_variants = data.get('num_variants', 3)  # Always 3 variants with OpenRouter

        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Validate service type (only openrouter allowed now)
        if translation_service_type != 'openrouter':
            return jsonify({
                'error': 'Invalid translation_service',
                'detail': f'Only "openrouter" is supported, got: {translation_service_type}',
                'suggestion': 'DeepL is currently disabled. Using OpenRouter for all translations.'
            }), 400

        # Check text length - allow longer text for PDFs but with reasonable limit
        # ~15k chars = ~3750 tokens input, leaves room for output with buffer
        max_chars = 15000  # Increased to support PDF translations
        chunk_size = 10000  # Optimal chunk size for translation quality
        
        # Check if we should auto-chunk (for very long texts)
        auto_chunk = data.get('auto_chunk', True)  # Default to True
        
        if len(text) > max_chars:
            if auto_chunk:
                # Automatically chunk and translate
                logger.info(f"Auto-chunking text: {len(text)} chars into chunks of {chunk_size}")
                chunks = []
                
                # Smart chunking: try to split on sentence boundaries
                sentences = text.replace('.\n', '.|NEWLINE|').replace('. ', '.|SPACE|').split('.')
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.replace('|NEWLINE|', '.\n').replace('|SPACE|', '. ')
                    if not sentence.strip():
                        continue
                        
                    # Add period back if it's not the last sentence
                    sentence = sentence + '.' if sentence.strip() else sentence
                    
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += sentence
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                logger.info(f"Split into {len(chunks)} chunks")
                
                # Translate each chunk
                all_chunk_translations = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                    try:
                        translation_service = get_prompting_service(service_type=translation_service_type)
                        chunk_result = translation_service.translate(
                            text=chunk,
                            source_lang=source_language,
                            target_lang=target_language,
                            domain=domain,
                            num_variants=1  # Only 1 variant per chunk to save credits/time
                        )
                        
                        chunk_translations = chunk_result.get('translations', [])
                        if chunk_translations:
                            all_chunk_translations.append(chunk_translations[0])
                        else:
                            all_chunk_translations.append(f"[Translation failed for chunk {i+1}]")
                    
                    except Exception as chunk_error:
                        error_msg = str(chunk_error)
                        logger.error(f"Chunk {i+1} translation failed with {translation_service_type}: {chunk_error}")
                        
                        # No fallback - DeepL is disabled
                        all_chunk_translations.append(f"[Translation failed for chunk {i+1}]")
                
                # Combine all chunk translations
                combined_translation = '\n\n'.join(all_chunk_translations)
                
                response = {
                    'original_text': text[:500] + '...' if len(text) > 500 else text,  # Truncate in response
                    'translation': combined_translation,
                    'translations': [combined_translation],
                    'source_language': source_language,
                    'target_language': target_language,
                    'domain': domain,
                    'timestamp': datetime.utcnow().isoformat(),
                    'variants': [combined_translation],
                    'metadata': {
                        'service': translation_service_type,
                        'chunked': True,
                        'num_chunks': len(chunks),
                        'original_length': len(text),
                        'translated_length': len(combined_translation)
                    }
                }
                
                logger.info(f"Chunked translation successful: {len(chunks)} chunks")
                return jsonify(response)
            else:
                # Don't auto-chunk, return error
                return jsonify({
                    'error': f'Text too long (max {max_chars} characters)',
                    'detail': f'Your text has {len(text)} characters. Enable auto_chunk or split manually.',
                    'suggestion': 'Set "auto_chunk": true in the request to automatically split and translate long texts.'
                }), 400

        # Use the configured translation service (DeepL by default)
        try:
            translation_service = get_prompting_service(service_type=translation_service_type)
            result = translation_service.translate(
                text=text,
                source_lang=source_language,
                target_lang=target_language,
                domain=domain,
                num_variants=num_variants  # DeepL=1, OpenRouter=customizable (default 3)
            )
            
            translations = result.get('translations', [])
            translation = translations[0] if translations else ''
            service_used = result.get('service', translation_service_type)
            
            # Return the translation response
            response = {
                'original_text': text,
                'translation': translation,
                'translations': translations,
                'source_language': source_language,
                'target_language': target_language,
                'domain': domain,
                'timestamp': datetime.utcnow().isoformat(),
                'variants': translations, # Kept for frontend compatibility
                'metadata': {
                    'service': service_used,
                    'num_variants': len(translations)
                }
            }
            
            logger.info(f"Translation successful ({service_used}): {len(translations)} variant(s) generated")
            return jsonify(response)
            
        except Exception as service_error:
            error_msg = str(service_error)
            logger.error(f"Translation service ({translation_service_type}) failed: {service_error}")
            
            # No fallback - DeepL is disabled, return error
            return jsonify({
                'error': 'Translation service failed',
                'detail': error_msg,
                'service': translation_service_type
            }), 500
            # Final fallback to mock translation if both services fail
            logger.warning("All translation services failed, using mock translation")
            translations = mock_translate(text, source_language, target_language, n=3)
            translation = translations[0] if translations else ''
            
            response = {
                'original_text': text,
                'translation': translation,
                'translations': translations,
                'source_language': source_language,
                'target_language': target_language,
                'timestamp': datetime.utcnow().isoformat(),
                'variants': translations,
                'warning': 'Translation service unavailable, using fallback mock translation.'
            }
            
            return jsonify(response)

    except Exception as e:
        logger.error(f"Translation endpoint error: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@api_bp.route('/upload-file', methods=['POST'])
def upload_file():
    """Accepts a single file upload (multipart/form-data) and extracts text from it.

    Supported types: .txt, .pdf, .docx
    Returns: { filename, content_type, text }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename
        content_type = file.content_type or ''
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        # Read bytes so we can reuse for different parsers
        file_bytes = file.read()

        text = ''
        if ext in ('txt', 'md', 'csv'):
            try:
                text = file_bytes.decode('utf-8')
            except Exception:
                text = file_bytes.decode('latin-1', errors='ignore')

        elif ext == 'pdf':
            # Quick sanity checks on the uploaded bytes
            if len(file_bytes) < 100:
                logger.warning('Uploaded PDF file is too small')
                return jsonify({'error': 'Invalid or empty PDF file', 'detail': 'Uploaded file is too small; it may be corrupt or empty.'}), 422
            
            if b'%PDF' not in file_bytes[:1024]:
                logger.warning('Uploaded file does not have PDF header')
                return jsonify({'error': 'Invalid PDF file', 'detail': 'File does not appear to be a valid PDF (missing %PDF header).'}), 422

            # Try PyMuPDF first (prefer import of 'pymupdf' to avoid naming conflicts)
            pdf_extraction_success = False
            text = ''
            
            try:
                try:
                    import pymupdf as fitz  # prefer the proper pymupdf package
                except ImportError:
                    import fitz  # fallback to 'fitz' package name
                
                logger.debug('Using PyMuPDF/fitz for PDF extraction')
                
                try:
                    # Try opening; catch errors to return clear response
                    doc = fitz.open(stream=file_bytes, filetype='pdf')
                except Exception as open_err:
                    logger.error(f'fitz failed to open PDF: {open_err}', exc_info=True)
                    return jsonify({'error': 'Failed to open PDF', 'detail': f'PDF file may be corrupted or password-protected: {str(open_err)}'}), 400

                pages_text = []
                total_chars = 0
                
                for i, page in enumerate(doc):
                    try:
                        # Extract text with different methods for better results
                        p_text = page.get_text('text') or ''
                        
                        # If no text extracted, try alternative methods
                        if not p_text.strip():
                            # Try blocks method
                            try:
                                blocks = page.get_text('blocks')
                                p_text = ' '.join([b[4] for b in blocks if len(b) >= 5 and b[4].strip()])
                            except:
                                pass
                        
                        # Clean up the extracted text
                        if p_text:
                            # Remove excessive whitespace while preserving line breaks
                            lines = [line.strip() for line in p_text.split('\n')]
                            p_text = '\n'.join([line for line in lines if line])
                        
                        pages_text.append(p_text)
                        total_chars += len(p_text)
                        logger.debug(f'Page {i+1}: extracted {len(p_text)} chars')
                    except Exception as perr:
                        logger.warning(f'page.get_text error on page {i+1}: {perr}')
                        pages_text.append('')

                # Join pages with double newline for better readability
                text = '\n\n'.join([p for p in pages_text if p.strip()])
                
                logger.info(f'Extracted total {len(text)} characters from PDF using PyMuPDF/fitz (pages: {doc.page_count})')
                pdf_extraction_success = True
                doc.close()

            except ImportError as import_err:
                logger.debug(f'PyMuPDF/fitz not available: {import_err}')
            except Exception as fitz_err:
                logger.warning(f'PyMuPDF/fitz extraction failed: {fitz_err}', exc_info=True)
            
            # Try pypdf fallback if PyMuPDF failed or is not available
            if not pdf_extraction_success:
                try:
                    from pypdf import PdfReader
                    import io
                    
                    logger.debug('Trying pypdf fallback for PDF extraction')
                    reader = PdfReader(io.BytesIO(file_bytes))
                    pages_text = []
                    
                    for i, page in enumerate(reader.pages):
                        try:
                            p_text = page.extract_text() or ''
                            # Clean up text
                            if p_text:
                                lines = [line.strip() for line in p_text.split('\n')]
                                p_text = '\n'.join([line for line in lines if line])
                            pages_text.append(p_text)
                            logger.debug(f'Page {i+1}: extracted {len(p_text)} chars')
                        except Exception as perr:
                            logger.warning(f'pypdf extract error on page {i+1}: {perr}')
                            pages_text.append('')
                    
                    text = '\n\n'.join([p for p in pages_text if p.strip()])
                    logger.info(f'Extracted total {len(text)} characters from PDF using pypdf (pages: {len(reader.pages)})')
                    pdf_extraction_success = True
                    
                except ImportError as import_err:
                    logger.error(f'pypdf not available: {import_err}')
                except Exception as pypdf_err:
                    logger.error(f'pypdf extraction failed: {pypdf_err}', exc_info=True)
            
            # Check if extraction was successful
            if not pdf_extraction_success:
                return jsonify({
                    'error': 'Failed to extract text from PDF',
                    'detail': 'Could not extract text. Ensure PyMuPDF (recommended) or pypdf is installed: pip install pymupdf pypdf'
                }), 500
            
            # Check if any text was extracted
            if not text.strip():
                logger.warning('PDF parsed successfully but no extractable text found')
                return jsonify({
                    'error': 'No extractable text found in PDF',
                    'detail': 'The PDF may be:\n1. A scanned image without embedded text (needs OCR)\n2. Empty\n3. Contains only images/graphics\n\nConsider using OCR software or uploading a text-based PDF.'
                }), 422

        elif ext in ('docx', 'doc'):
            try:
                from docx import Document
                import io
                doc = Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text]
                text = '\n'.join(paragraphs)
            except Exception as e:
                logger.error(f'DOCX extraction failed: {e}', exc_info=True)
                return jsonify({'error': 'Failed to extract text from Word document'}), 500

        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

        return jsonify({'filename': filename, 'content_type': content_type, 'text': text}), 200

    except Exception as e:
        logger.error(f'File upload/extract error: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error while extracting file'}), 500

@api_bp.route('/detect-language', methods=['POST'])
def detect_language():
    """Language detection endpoint"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400

        text = data['text'].strip()

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Simple mock language detection
        # In production, use a proper language detection library
        if any(char in 'abcdefghijklmnopqrstuvwxyz' for char in text.lower()):
            detected_lang = 'en' if text[0].isupper() else 'en'  # Very basic
        elif any(char in 'àâäéèêëïîôöùûüÿç' for char in text.lower()):
            detected_lang = 'fr'
        else:
            detected_lang = 'ar'  # Default to Arabic for this project

        return jsonify({
            'text': text,
            'detected_language': detected_lang,
            'confidence': 0.8  # Mock confidence
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@api_bp.route('/submit-preferences', methods=['POST'])
def submit_preferences():
    """
    Store user translation preferences in a CSV file.
    Each pair comparison between translations is stored as a separate row.
    
    Expected data:
    {
        "sessionId": "unique-session-id",
        "sourceText": "original text",
        "sourceLanguage": "en",
        "targetLanguage": "ar",
        "rankings": [
            {"variantId": "id1", "rank": 1, "text": "translation 1", "isCustom": false, "isEdited": false},
            {"variantId": "id2", "rank": 2, "text": "translation 2", "isCustom": false, "isEdited": false},
            ...
        ],
        "timestamp": "2026-01-01T12:00:00Z"
    }
    """
    import csv
    import os
    from datetime import datetime
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract data
        session_id = data.get('sessionId', 'unknown')
        source_text = data.get('sourceText', '')
        source_language = data.get('sourceLanguage', 'en')
        target_language = data.get('targetLanguage', 'ar')
        rankings = data.get('rankings', [])
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        
        if not rankings or len(rankings) < 2:
            return jsonify({'error': 'At least 2 translations required for preference comparison'}), 400
        
        # Create preferences directory if it doesn't exist
        preferences_dir = os.path.join(os.path.dirname(__file__), '..', 'user_preferences')
        os.makedirs(preferences_dir, exist_ok=True)
        
        # CSV file path
        csv_file = os.path.join(preferences_dir, 'translation_preferences.csv')
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_file)
        
        # Open CSV file in append mode
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp',
                'session_id',
                'source_text',
                'source_language',
                'target_language',
                'preferred_translation',
                'preferred_is_custom',
                'preferred_is_edited',
                'compared_translation',
                'compared_is_custom',
                'compared_is_edited'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Create pairwise comparisons based on user ordering
            # For each translation, compare it with all translations that come after it
            # The order in the rankings list determines preference (earlier = better)
            for i, preferred in enumerate(rankings):
                for j, compared in enumerate(rankings):
                    if i < j:  # Only compare with translations that come after
                        row = {
                            'timestamp': timestamp,
                            'session_id': session_id,
                            'source_text': source_text[:500] if len(source_text) > 500 else source_text,  # Truncate long texts
                            'source_language': source_language,
                            'target_language': target_language,
                            'preferred_translation': preferred['text'],
                            'preferred_is_custom': preferred.get('isCustom', False),
                            'preferred_is_edited': preferred.get('isEdited', False),
                            'compared_translation': compared['text'],
                            'compared_is_custom': compared.get('isCustom', False),
                            'compared_is_edited': compared.get('isEdited', False)
                        }
                        
                        writer.writerow(row)
            
            logger.info(f"Stored {len(rankings) * (len(rankings) - 1) // 2} preference pairs for session {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Preferences stored successfully',
            'pairs_stored': len(rankings) * (len(rankings) - 1) // 2,
            'file_path': csv_file
        }), 200
        
    except Exception as e:
        logger.error(f"Error storing preferences: {e}", exc_info=True)
        return jsonify({'error': f'Failed to store preferences: {str(e)}'}), 500