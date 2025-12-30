# WindGrib Dependencies Summary

## 📋 Corrections et Améliorations Apportées

### 1. **Fichier `requirements.txt` corrigé**

**Problème** : Le fichier ne contenait que des dépendances de développement et manquait les dépendances principales nécessaires pour faire fonctionner le package.

**Solution** : 
- Ajout des dépendances principales requises par le package
- Organisation claire en sections commentées
- Ajout des versions minimales requises

### 2. **Nouveau fichier `requirements-dev.txt` créé**

**Objectif** : Séparer les dépendances de développement des dépendances principales pour une meilleure gestion.

**Contenu** :
- Dépendances de développement (linting, testing, documentation)
- Outils de build et publication
- Dépendances optionnelles pour le développement

### 3. **Fichier `pyproject.toml` amélioré**

**Améliorations** :
- Ajout des dépendances optionnelles (`[project.optional-dependencies]`)
- Ajout de classificateurs supplémentaires pour les versions Python
- Ajout de mots-clés supplémentaires pour une meilleure découvrabilité
- Correction de la duplication des dépendances

### 4. **Documentation mise à jour**

**Améliorations** :
- Instructions d'installation plus complètes
- Séparation des instructions pour différents cas d'usage
- Ajout d'exemples pour l'installation en développement

## 📦 Structure des Dépendances

### Dépendances Principales (requises)
```
numpy>=1.20.0
pandas>=1.3.0
xarray>=0.20.0
s3fs>=2021.11.0
requests>=2.26.0
tqdm>=4.62.0
cfgrib>=0.9.10.0
dask>=2021.11.0
```

### Dépendances de Développement (optionnelles)
```
eccodes>=0.9.8
black>=23.12.1
isort>=5.13.2
mypy>=1.8.0
pytest>=7.4.3
pytest-cov>=4.1.0
build>=1.0.3
twine>=4.0.2
types-requests
mkdocs>=1.5.3
mkdocs-material>=9.5.3
flake8>=6.1.0
pylint>=3.2.7
coverage>=7.6.1
pytest-mock>=3.14.0
```

## 🚀 Méthodes d'Installation

### 1. Installation basique (utilisateur final)
```bash
pip install windgrib
```

### 2. Installation depuis le source
```bash
pip install .
```

### 3. Installation en mode développement
```bash
pip install -e .
```

### 4. Installation avec dépendances de développement
```bash
pip install -e ".[dev]"
# ou
pip install -r requirements-dev.txt
```

## ✅ Vérification des Dépendances

Pour vérifier que toutes les dépendances sont correctement installées :

```python
import windgrib
from windgrib import Grib
print("Version:", windgrib.__version__)
print("All dependencies working!")
```

## 🔧 Gestion des Dépendances

### Ajouter une nouvelle dépendance principale
1. Ajouter à `pyproject.toml` dans la section `dependencies`
2. Ajouter à `requirements.txt` dans la section principale
3. Mettre à jour la documentation si nécessaire

### Ajouter une dépendance de développement
1. Ajouter à `pyproject.toml` dans la section `[project.optional-dependencies]`
2. Ajouter à `requirements-dev.txt`
3. Ajouter à `requirements.txt` pour la compatibilité

## 📝 Notes Importantes

- Les dépendances dans `pyproject.toml` sont prioritaires pour la publication sur PyPI
- `requirements.txt` est principalement pour les développeurs et la compatibilité
- `requirements-dev.txt` est pour le développement local
- Les versions minimales sont spécifiées pour assurer la compatibilité

## 🎯 Prochaines Étapes

1. **Tester l'installation depuis PyPI** (quand publié)
2. **Vérifier les dépendances dans différents environnements**
3. **Mettre à jour les dépendances régulièrement**
4. **Documenter les changements de dépendances** dans le CHANGELOG

---

*Dernière mise à jour : 30 décembre 2025*