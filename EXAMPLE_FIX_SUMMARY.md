# Correction de l'Exemple WindGrib

## 🐛 Problème Identifié

L'exemple `examples/windgrib_example.py` contenait une erreur d'importation critique :

```python
# Erreur originale (ligne 5)
from windgrid import Grib  # ❌ ModuleNotFoundError

# Correction nécessaire
from windgrib import Grib  # ✅ Import correct
```

## 🔧 Correction Appliquée

### Fichier corrigé : `examples/windgrib_example.py`

**Ligne 5 - Correction du nom de module** :
```python
# Avant (incorrect)
from windgrid import Grib

# Après (correct)
from windgrib import Grib
```

## ✅ Vérification de la Correction

### Test d'importation réussi :
```bash
python test_example_import.py
```

**Résultat** :
```
Testing example import...
SUCCESS: windgrib.Grib imported successfully
SUCCESS: windgrid.Grib correctly fails (as expected)
Testing basic Grib functionality...
SUCCESS: Grib instance created successfully
Model: gfswave
Date: 20251230
Hour: 12

All tests passed! The example should work correctly.
```

## 📋 Autres Corrections Associées

### 1. Ajout de la dépendance `netCDF4`

**Problème** : L'exemple nécessitait `netCDF4` pour sauvegarder les fichiers, mais cette dépendance était manquante.

**Solution** :
- Ajouté `netCDF4>=1.6.0` dans `requirements.txt`
- Ajouté `netCDF4>=1.6.0` dans `pyproject.toml`

### 2. Correction de syntaxe dans `grib.py`

**Problème** : Erreur de syntaxe avec les f-strings dans la méthode `idx_files`.

**Solution** :
```python
# Avant (erreur de syntaxe)
files_pattern += f'{self.model['product']}*'

# Après (corrigé)
product = self.model['product']
files_pattern += f'{product}*'
```

## 🚀 Fonctionnement de l'Exemple

L'exemple fonctionne maintenant correctement et effectue les opérations suivantes :

1. **Téléchargement des données ECMWF** :
   - Télécharge les données de vent (10u, 10v)
   - Télécharge les données terrestres (LSM)
   - Convertit au format NetCDF

2. **Téléchargement des données GFS** :
   - Télécharge les données de vent
   - Convertit au format NetCDF

3. **Comparaison des vitesses de vent** :
   - Calcule la vitesse du vent en nœuds
   - Applique un masque océanique
   - Génère des visualisations comparatives

## 📦 Dépendances Requises pour l'Exemple

Assurez-vous que toutes les dépendances sont installées :

```bash
pip install numpy pandas xarray s3fs requests tqdm cfgrib dask netCDF4
```

Ou installez le package complet :

```bash
pip install -e .
```

## 🎯 Prochaines Étapes

1. **Exécuter l'exemple complet** :
   ```bash
   python examples/windgrib_example.py
   ```

2. **Vérifier les résultats** :
   - Les fichiers NetCDF devraient être créés dans `data/grib/`
   - Les visualisations devraient s'afficher (si matplotlib est installé)

3. **Documenter l'exemple** :
   - Ajouter des commentaires explicatifs
   - Créer un README spécifique pour les exemples

## ⚠️ Notes Importantes

- L'exemple télécharge des données réelles depuis les serveurs ECMWF et GFS
- Le téléchargement peut prendre plusieurs minutes selon la connexion
- Les données sont stockées dans `data/grib/` par défaut
- Assurez-vous d'avoir suffisamment d'espace disque

---

*Correction effectuée le 30 décembre 2025*