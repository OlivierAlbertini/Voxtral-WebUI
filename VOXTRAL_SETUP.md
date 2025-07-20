# Configuration de Voxtral-Mini

Ce projet a été modifié pour utiliser **Voxtral-Mini-3B-2507** comme modèle de transcription par défaut au lieu de faster-whisper.

## Installation des dépendances Voxtral

### Option 1: Installation automatique (Windows)
Exécutez le script fourni :
```batch
install-voxtral.bat
```

### Option 2: Installation manuelle

1. **Désinstaller l'ancienne version de transformers:**
```bash
pip uninstall transformers -y
```

2. **Installer transformers depuis la source (version développement):**
```bash
pip install git+https://github.com/huggingface/transformers.git
```

3. **Installer les dépendances spécifiques à Voxtral:**
```bash
pip install mistral-common[audio]
pip install librosa
pip install soundfile
```

### Option 3: Utiliser faster-whisper en fallback

Si vous ne pouvez pas installer Voxtral, le système utilisera automatiquement faster-whisper comme fallback avec un avertissement.

Pour forcer l'utilisation de faster-whisper, lancez l'application avec :
```bash
python app.py --whisper_type faster-whisper
```

## Fonctionnalités Voxtral

- **Modèle multilingue** supportant 8 langues
- **Détection automatique de langue**
- **Performance optimisée** pour GPU CUDA
- **Contexte étendu** jusqu'à 30 minutes d'audio
- **Intégration complète** avec VAD, diarisation et séparation BGM

## Configuration

Le modèle Voxtral sera téléchargé automatiquement depuis Hugging Face lors de la première utilisation.

**Espace requis:** ~9.5 GB de VRAM GPU pour des performances optimales.

## Dépannage

### Erreur ImportError VoxtralForConditionalGeneration
Cette erreur indique que votre version de transformers ne supporte pas encore Voxtral. Suivez l'installation manuelle ci-dessus.

### Mémoire insuffisante
Si vous manquez de VRAM, utilisez faster-whisper à la place :
```bash
python app.py --whisper_type faster-whisper
```

### Problèmes de dépendances audio
Assurez-vous que `librosa` et `soundfile` sont correctement installés :
```bash
pip install librosa soundfile
```