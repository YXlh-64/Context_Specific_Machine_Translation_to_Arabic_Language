"""
Script to generate diverse, high-quality test prompts for synthetic data generation.

This script creates prompts covering multiple domains, lengths, and complexities
to ensure the reward model learns from diverse translation scenarios.
"""

import json
import random

# Domain-specific prompt templates
PROMPT_CATEGORIES = {
    "greetings": {
        "en": [
            "Hello, how are you?",
            "Good morning, have a great day!",
            "Nice to meet you.",
            "How have you been lately?",
            "Welcome to our store.",
            "It's a pleasure to see you again.",
            "Hi there! How can I help you today?",
        ],
        "fr": [
            "Bonjour, comment allez-vous?",
            "Bonne journée!",
            "Enchanté de vous rencontrer.",
            "Comment avez-vous été récemment?",
            "Bienvenue dans notre magasin.",
            "C'est un plaisir de vous revoir.",
            "Salut! Comment puis-je vous aider aujourd'hui?",
        ]
    },
    
    "travel": {
        "en": [
            "Where is the nearest airport?",
            "I need directions to the train station.",
            "How much does a taxi to downtown cost?",
            "Is there a bus that goes to the museum?",
            "Can you recommend a good hotel nearby?",
            "What time does the next flight depart?",
            "I would like to book a room for two nights.",
            "Are there any tours available tomorrow?",
        ],
        "fr": [
            "Où se trouve l'aéroport le plus proche?",
            "J'ai besoin d'indications pour la gare.",
            "Combien coûte un taxi pour le centre-ville?",
            "Y a-t-il un bus qui va au musée?",
            "Pouvez-vous recommander un bon hôtel à proximité?",
            "À quelle heure part le prochain vol?",
            "Je voudrais réserver une chambre pour deux nuits.",
            "Y a-t-il des visites guidées disponibles demain?",
        ]
    },
    
    "medical": {
        "en": [
            "I need to see a doctor immediately.",
            "Where is the nearest hospital?",
            "I have a severe headache and fever.",
            "Can you call an ambulance please?",
            "I'm allergic to penicillin.",
            "What are the visiting hours?",
            "I need a prescription refill.",
            "Do you accept my health insurance?",
        ],
        "fr": [
            "J'ai besoin de voir un médecin immédiatement.",
            "Où se trouve l'hôpital le plus proche?",
            "J'ai un mal de tête sévère et de la fièvre.",
            "Pouvez-vous appeler une ambulance s'il vous plaît?",
            "Je suis allergique à la pénicilline.",
            "Quelles sont les heures de visite?",
            "J'ai besoin d'un renouvellement d'ordonnance.",
            "Acceptez-vous mon assurance santé?",
        ]
    },
    
    "shopping": {
        "en": [
            "How much does this cost?",
            "Can I try this on?",
            "Do you have this in a different size?",
            "Can I pay by credit card?",
            "Is there a discount for bulk purchases?",
            "What is your return policy?",
            "I would like to return this item.",
            "Are these products on sale?",
        ],
        "fr": [
            "Combien ça coûte?",
            "Puis-je l'essayer?",
            "Avez-vous ceci dans une autre taille?",
            "Puis-je payer par carte de crédit?",
            "Y a-t-il une réduction pour les achats en gros?",
            "Quelle est votre politique de retour?",
            "Je voudrais retourner cet article.",
            "Ces produits sont-ils en solde?",
        ]
    },
    
    "restaurant": {
        "en": [
            "Can I see the menu please?",
            "I would like to order the grilled chicken.",
            "Is this dish spicy?",
            "I'm vegetarian, what do you recommend?",
            "Could I have the bill please?",
            "Do you have any gluten-free options?",
            "This meal was delicious, thank you!",
            "Can I make a reservation for four people?",
        ],
        "fr": [
            "Puis-je voir le menu s'il vous plaît?",
            "Je voudrais commander le poulet grillé.",
            "Ce plat est-il épicé?",
            "Je suis végétarien, que recommandez-vous?",
            "Pourrais-je avoir l'addition s'il vous plaît?",
            "Avez-vous des options sans gluten?",
            "Ce repas était délicieux, merci!",
            "Puis-je faire une réservation pour quatre personnes?",
        ]
    },
    
    "business": {
        "en": [
            "We need to schedule a meeting for next week.",
            "Please send me the quarterly report by Friday.",
            "I would like to discuss the project timeline.",
            "Can you confirm the delivery date?",
            "We are interested in your product catalog.",
            "What are your payment terms?",
            "I will review the contract and get back to you.",
            "Let's arrange a conference call with the team.",
        ],
        "fr": [
            "Nous devons planifier une réunion pour la semaine prochaine.",
            "Veuillez m'envoyer le rapport trimestriel d'ici vendredi.",
            "Je voudrais discuter du calendrier du projet.",
            "Pouvez-vous confirmer la date de livraison?",
            "Nous sommes intéressés par votre catalogue de produits.",
            "Quelles sont vos conditions de paiement?",
            "Je vais examiner le contrat et vous recontacterai.",
            "Organisons une conférence téléphonique avec l'équipe.",
        ]
    },
    
    "education": {
        "en": [
            "What are the admission requirements?",
            "When does the semester start?",
            "I would like to register for this course.",
            "Can you explain this concept again?",
            "Where is the library located?",
            "What time is the exam?",
            "I need help with my homework.",
            "Are scholarships available for international students?",
        ],
        "fr": [
            "Quelles sont les conditions d'admission?",
            "Quand commence le semestre?",
            "Je voudrais m'inscrire à ce cours.",
            "Pouvez-vous expliquer ce concept à nouveau?",
            "Où se trouve la bibliothèque?",
            "À quelle heure est l'examen?",
            "J'ai besoin d'aide pour mes devoirs.",
            "Des bourses sont-elles disponibles pour les étudiants internationaux?",
        ]
    },
    
    "technology": {
        "en": [
            "My computer won't start.",
            "How do I reset my password?",
            "The application keeps crashing.",
            "Can you help me install this software?",
            "I need to upgrade my internet connection.",
            "What are the system requirements?",
            "Is this device compatible with my phone?",
            "How do I back up my data?",
        ],
        "fr": [
            "Mon ordinateur ne démarre pas.",
            "Comment réinitialiser mon mot de passe?",
            "L'application continue de planter.",
            "Pouvez-vous m'aider à installer ce logiciel?",
            "J'ai besoin de mettre à niveau ma connexion internet.",
            "Quelles sont les exigences système?",
            "Cet appareil est-il compatible avec mon téléphone?",
            "Comment sauvegarder mes données?",
        ]
    },
    
    "complex_sentences": {
        "en": [
            "Although the weather forecast predicted rain, we decided to go hiking because we had planned this trip for months.",
            "The research team, which has been studying climate change for over a decade, published their findings in a prestigious scientific journal.",
            "If you could please send me the updated financial projections along with the marketing strategy document, I would greatly appreciate it.",
            "The company announced that it would be implementing new sustainability initiatives while also expanding its operations into emerging markets.",
            "Despite facing numerous challenges throughout the project, the development team successfully delivered the software on time and within budget.",
        ],
        "fr": [
            "Bien que les prévisions météorologiques annonçaient de la pluie, nous avons décidé de faire de la randonnée car nous avions prévu ce voyage depuis des mois.",
            "L'équipe de recherche, qui étudie le changement climatique depuis plus d'une décennie, a publié ses résultats dans une revue scientifique prestigieuse.",
            "Si vous pouviez m'envoyer les projections financières mises à jour ainsi que le document de stratégie marketing, je vous en serais très reconnaissant.",
            "L'entreprise a annoncé qu'elle mettrait en œuvre de nouvelles initiatives de durabilité tout en élargissant ses opérations sur les marchés émergents.",
            "Malgré de nombreux défis tout au long du projet, l'équipe de développement a réussi à livrer le logiciel dans les délais et dans les limites du budget.",
        ]
    }
}


def generate_diverse_prompts(num_prompts_per_category=10):
    """Generate a diverse set of test prompts"""
    
    prompts = []
    
    for category, languages in PROMPT_CATEGORIES.items():
        for lang, sentences in languages.items():
            # Use all sentences in each category
            for sentence in sentences:
                prompts.append({
                    "text": sentence,
                    "lang": lang,
                    "category": category,
                    "length": len(sentence.split())
                })
    
    # Shuffle to mix categories and languages
    random.shuffle(prompts)
    
    return prompts


def save_prompts(prompts, output_file):
    """Save prompts to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            # Write only text and lang for the actual use
            # Keep category and length as metadata (commented or in separate field)
            json.dump({
                "text": prompt["text"],
                "lang": prompt["lang"],
                # Optionally include metadata
                "metadata": {
                    "category": prompt["category"],
                    "word_count": prompt["length"]
                }
            }, f, ensure_ascii=False)
            f.write('\n')


def print_statistics(prompts):
    """Print statistics about the generated prompts"""
    
    print("\n" + "="*80)
    print("PROMPT DATASET STATISTICS")
    print("="*80)
    
    print(f"\nTotal prompts: {len(prompts)}")
    
    # By language
    lang_counts = {}
    for p in prompts:
        lang_counts[p['lang']] = lang_counts.get(p['lang'], 0) + 1
    print(f"\nBy Language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang.upper()}: {count} prompts")
    
    # By category
    cat_counts = {}
    for p in prompts:
        cat_counts[p['category']] = cat_counts.get(p['category'], 0) + 1
    print(f"\nBy Category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} prompts")
    
    # By length
    lengths = [p['length'] for p in prompts]
    print(f"\nLength Statistics:")
    print(f"  Min: {min(lengths)} words")
    print(f"  Max: {max(lengths)} words")
    print(f"  Average: {sum(lengths)/len(lengths):.1f} words")
    
    # Length distribution
    short = sum(1 for l in lengths if l < 10)
    medium = sum(1 for l in lengths if 10 <= l < 20)
    long_text = sum(1 for l in lengths if l >= 20)
    print(f"\nLength Distribution:")
    print(f"  Short (<10 words): {short}")
    print(f"  Medium (10-19 words): {medium}")
    print(f"  Long (≥20 words): {long_text}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Generate prompts
    prompts = generate_diverse_prompts()
    
    # Print statistics
    print_statistics(prompts)
    
    # Save to file
    output_file = "test_prompts_diverse.jsonl"
    save_prompts(prompts, output_file)
    
    print(f"\n✓ Saved {len(prompts)} prompts to: {output_file}")
    print("\nSample prompts:")
    for i, p in enumerate(random.sample(prompts, min(5, len(prompts))), 1):
        print(f"\n{i}. [{p['lang'].upper()}] {p['category']}")
        print(f"   {p['text'][:80]}...")
