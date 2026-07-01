# TFM - Detección de Retinopatía Diabética

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Ejecutar un solo experimento (rápido, para probar que todo funciona)

```bash
cd src
python train_cnn.py --architecture efficientnet_b0 --task_mode binary
```

Esto lanza automáticamente las `n_runs` (3 por defecto) con distintas seeds y guarda:
- Checkpoints en `outputs/checkpoints/`
- Resumen de métricas (media ± desviación) en `outputs/results/`

## Ejecutar la matriz completa (binario vs multiclase)

```bash
cd src
python run_all_experiments.py
```

Esto entrena las 2 arquitecturas (EfficientNet-B0, MobileNetV3-Large) en los 2 modos (binario, multiclase), 3 veces cada uno = 12 entrenamientos en total

Al final genera `outputs/results/comparativa_binario_vs_multiclase.csv`


## Notas

- `dataset.py` aplica CLAHE por defecto (configurable en `config.yaml`,
  apartado `augmentation.clahe`), hipótesis H4 del TFM
- El primer run tardará más por la descarga de pesos preentrenados de ImageNet (se cachean en `~/.cache/torch` tras la primera vez)
