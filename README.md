# Análisis de toma de decisión en el tiro: Bayer Leverkusen 2023/24

Este repositorio contiene el trabajo final de análisis de la toma de decisiones en el tiro del Bayer Leverkusen durante la Bundesliga 2023/24, utilizando datos abiertos de StatsBomb Open Data y, especialmente, datos StatsBomb 360.

El objetivo principal es evaluar si, en cada situación de tiro, el jugador tomó una buena decisión o si existía un compañero mejor colocado con una línea de pase suficientemente favorable.

## Objetivo del proyecto

El trabajo busca responder a la pregunta:

> ¿En una acción de tiro, era mejor finalizar o pasar a un compañero en mejor posición?

Para ello se calcula una métrica de decisión basada en:

- Probabilidad geométrica de marcar desde la posición del tirador.
- Posición de compañeros y rivales visibles en el freeze frame 360.
- Probabilidad posicional de interceptación del pase.
- Probabilidad de conservación de la posesión por parte del equipo atacante.
- Valor esperado de pasar frente a tirar.
- Contexto temporal y marcador mediante heatmaps minuto-marcador.
- Redes de pase previas al tiro y métricas de red.

## Estructura del repositorio

```text
bayer-leverkursen-toma-de-decision/
├─ data/
│  ├─ README.md
│  ├─ competitions.json
│  ├─ matches_9_281.json
│  ├─ bayer_leverkusen_matches.json
│  ├─ events/
│  ├─ three-sixty/
│  └─ lineups/
├─ doc/
├─ outputs/
├─ src/
│  ├─ Tiro.py
│  └─ passing_network.py
├─ requirements.txt
└─ README.md
```

## Datos

Los datos proceden de StatsBomb Open Data:

https://github.com/statsbomb/open-data

Competición utilizada:

- Bundesliga 2023/24
- `competition_id = 9`
- `season_id = 281`
- Equipo analizado: Bayer Leverkusen

La carpeta `data/` contiene o debe contener los archivos de eventos, alineaciones y datos 360 de los 34 partidos del Bayer Leverkusen en Bundesliga 2023/24.

Si la carpeta `data/` no se sube completa a GitHub por tamaño, debe reconstruirse localmente siguiendo las instrucciones de `data/README.md`.

## Codigo principal

`src/Tiro.py`

Script principal del análisis. Carga eventos y datos 360, filtra los tiros del Bayer Leverkusen, calcula la probabilidad de marcar, evalúa alternativas de pase y genera tablas/gráficos de resultados.

`src/passing_network.py`

Script complementario para el análisis de redes de pase previas al tiro. Calcula métricas de red y genera visualizaciones asociadas.

## Instalacion

Se recomienda crear un entorno virtual antes de instalar dependencias.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

Desde la raíz del proyecto:

```bash
python src/Tiro.py
```

Para el análisis de redes:

```bash
python src/passing_network.py
```

Los resultados se guardan en la carpeta `outputs/`.

## Resultados generados

La carpeta `outputs/` incluye, entre otros:

- `decision_results.csv`: resultados tiro a tiro.
- `summary_metrics.csv`: resumen global del analisis.
- `player_decision_summary.csv`: resumen por jugador.
- `player_ranking.png`: ranking visual por jugador.
- `heatmap_4ventanas_5marcadores.png`: heatmap minuto-marcador.
- `heatmap_decision_margin_4ventanas_5marcadores.png`: heatmap con decision margin.
- `network_metrics.csv`: métricas globales de redes de pase.
- `network_player_summary.csv`: resumen de red por jugador.
- `red_pases_ejemplo.png`: ejemplo visual de red de pases previa al tiro.

## Metodologia resumida

1. Se cargan eventos, alineaciones y datos 360 de StatsBomb.
2. Se seleccionan los tiros del Bayer Leverkusen con freeze frame 360 disponible.
3. Se calcula la probabilidad de marcar del tirador a partir de distancia, ángulo y oposición defensiva.
4. Se identifican compañeros visibles en posiciones válidas.
5. Se estima la probabilidad de que el pase sea interceptado por rivales desde una perspectiva posicional.
6. Se calcula el valor esperado de pasar a cada compañero.
7. Se compara el valor del mejor pase con el valor del tiro.
8. Se clasifica la decisión y se agregan los resultados por jugador, contexto de partido y redes.

## Nota sobre StatsBomb Open Data

Este proyecto utiliza datos abiertos de StatsBomb. Los datos pertenecen a StatsBomb y se emplean con fines académicos. Para cualquier reutilización, debe revisarse la licencia y condiciones de uso del repositorio oficial de StatsBomb Open Data.

## Autor

David Thomas Jacobs Maritzia

Máster Universitario en Análisis de Datos Deportivos  
Universidad Rey Juan Carlos
