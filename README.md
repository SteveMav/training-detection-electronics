# training-detection-electronics

Entrainement YOLO pour detecter des objets electroniques complets avec le dataset annote ElectroCom-61 exporte depuis Roboflow.

Le repo est concu pour Windows avec une carte NVIDIA, par exemple une GeForce RTX 4070. La commande principale est:

```powershell
.\start_training.ps1
```

## Ce que le script fait

1. cree `.venv` si necessaire
2. installe PyTorch CUDA, Ultralytics et les dependances Python
3. verifie que CUDA voit le GPU
4. verifie que le dataset existe
5. corrige automatiquement `data.yaml` si les chemins Roboflow sont du type `../train/images`
6. valide les images, labels YOLO, classes et splits
7. lance l'entrainement Ultralytics YOLO
8. ecrit les resultats dans `runs/detect/electrocom61-v1/`
9. copie le meilleur modele vers `models/electrocom61/best.pt`

Le dataset, les runs et les poids ne sont jamais suivis par Git.

## Prerequis

- Windows
- Python 3.10 ou plus recent
- Git
- GPU NVIDIA avec pilotes recents
- Connexion internet au premier lancement pour installer les dependances et telecharger `yolo11s.pt`

Si PowerShell bloque les scripts sur votre machine:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Installation

Clonez le repo:

```powershell
git clone https://github.com/SteveMav/training-detection-electronics.git
cd training-detection-electronics
```

Placez le dataset annote dans le projet:

```text
data_t/ElectroCom-61_v2/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

Puis lancez:

```powershell
.\start_training.ps1
```

## Modes disponibles

Mode V1 recommande, par defaut:

```powershell
.\start_training.ps1
```

Equivalent:

```powershell
.\start_training.ps1 -Preset recommended
```

Parametres:

- modele: `yolo11s.pt`
- epochs: `50`
- image size: `640`
- device: `0`
- batch: `auto`

Mode rapide:

```powershell
.\start_training.ps1 -Preset fast
```

Parametres:

- modele: `yolo11n.pt`
- epochs: `30`
- image size: `640`
- device: `0`
- batch: `auto`

Reprendre apres interruption:

```powershell
.\start_training.ps1 -Resume
```

Changer le dataset, le modele ou le nombre d'epochs:

```powershell
.\start_training.ps1 -Dataset "data_t/ElectroCom-61_v2" -Model yolo11s.pt -Epochs 50
```

## Sorties

Ultralytics ecrit les artefacts ici:

```text
runs/detect/electrocom61-v1/
```

Le meilleur poids est copie automatiquement ici:

```text
models/electrocom61/best.pt
```

Ces fichiers sont ignores par Git parce qu'ils sont volumineux et regenerables.

## Estimation RTX 4070

- `yolo11n.pt`, 30 epochs: environ 20 a 45 minutes
- `yolo11s.pt`, 50 epochs: environ 45 minutes a 2 heures

La duree depend des versions CUDA/PyTorch, du pilote, de la charge GPU et du debit disque.

## Validation du dataset seule

```powershell
python scripts/prepare_dataset.py --dataset data_t/ElectroCom-61_v2 --fix
```

Le script verifie notamment:

- presence de `data.yaml`
- chemins `train`, `val`, `test`
- presence des dossiers `images` et `labels`
- correspondance image/label
- format YOLO des labels
- indices de classes valides
- images lisibles

## Verification CUDA seule

```powershell
python scripts/check_cuda.py --require-cuda --device 0
```

## Depannage rapide

Si CUDA n'est pas detecte, verifiez d'abord le pilote NVIDIA:

```powershell
nvidia-smi
```

PyTorch est necessaire parce qu'Ultralytics entraine YOLO avec le backend PyTorch. Le script l'installe separement pour forcer un build CUDA adapte au GPU, puis installe les autres dependances depuis `requirements.txt`.

Index CUDA utilise par defaut:

```text
https://download.pytorch.org/whl/cu121
```

Vous pouvez le remplacer:

```powershell
.\start_training.ps1 -TorchIndexUrl "https://download.pytorch.org/whl/cu126"
```

Si PyTorch CUDA est deja correctement installe dans `.venv`:

```powershell
.\start_training.ps1 -SkipTorchInstall
```
