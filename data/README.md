# Datos del proyecto

Esta carpeta contiene los datos de StatsBomb Open Data utilizados para el analisis de la toma de decision en el tiro del Bayer Leverkusen durante la Bundesliga 2023/24.

Por tamano, los archivos JSON completos pueden no estar incluidos en el repositorio. Para ejecutar el proyecto localmente, la carpeta `data/` debe reconstruirse con la siguiente estructura:

```text
data/
├─ competitions.json
├─ matches_9_281.json
├─ bayer_leverkusen_matches.json
├─ events/
├─ three-sixty/
└─ lineups/
```

## Origen de los datos

Los datos proceden de StatsBomb Open Data:

https://github.com/statsbomb/open-data

La competicion utilizada es:

- Bundesliga 2023/24
- `competition_id = 9`
- `season_id = 281`

## Contenido esperado

`competitions.json`  
Listado general de competiciones disponibles en StatsBomb Open Data.

`matches_9_281.json`  
Partidos de la Bundesliga 2023/24.

`bayer_leverkusen_matches.json`  
Subconjunto de partidos relacionados con el Bayer Leverkusen en Bundesliga 2023/24. Es el archivo de referencia utilizado por el codigo para seleccionar los partidos del analisis.

`events/`  
Eventos de partido en formato StatsBomb. Incluye tiros, pases, conducciones, faltas, recuperaciones y el resto de acciones registradas.

`three-sixty/`  
Datos StatsBomb 360. Contienen los frames visibles alrededor de cada evento, con posiciones de jugadores visibles, companeros, rivales y portero. Son los datos centrales del trabajo porque permiten evaluar la toma de decision con informacion espacial.

`lineups/`  
Alineaciones y jugadores disponibles por partido. Se utilizan para identificar plantillas, equipos y nombres de jugadores.

## Archivos necesarios para ejecutar el codigo

Para que `src/Tiro.py` funcione correctamente deben existir:

- `data/bayer_leverkusen_matches.json`
- `data/events/*.json`
- `data/three-sixty/*.json`
- `data/lineups/*.json`

En este trabajo se esperan 34 archivos por carpeta en:

- `data/events/`
- `data/three-sixty/`
- `data/lineups/`

## Nota sobre GitHub

Si GitHub no permite subir la carpeta `data/` completa por tamano, se recomienda dejar este `README.md` en el repositorio y mantener los datos JSON localmente. La ejecucion del proyecto requiere descargar o copiar de nuevo los datos desde StatsBomb Open Data siguiendo la estructura indicada.
