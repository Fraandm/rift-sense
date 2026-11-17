# RiftSense · Analizador de draft y build para League of Legends

Sistema completo de análisis de partidas de LoL: descarga un dataset grande de
partidas profesionales, entrena dos redes neuronales y expone una interfaz web
responsive para recomendar campeones y objetos.

```
Leaguepedia / Riot API  →  dataset  →  DraftNet (transformer)  ┐
Data Dragon (estático)  →  priors   →  BuildNet (GRU)          ├→  FastAPI  →  SPA
                                                                ┘
```

> Datos reales, métricas reales y límites explicados. No es una bola de cristal.

---

## 1. Qué incluye

| Pieza | Descripción |
| --- | --- |
| **Scraper pro** | API Cargo de Leaguepedia. Reanudable, con backoff ante rate limits y export masivo de 5.000 filas por petición. |
| **Scraper high elo (opcional)** | Riot API: Challenger/GM/Master, match-v5 + timeline. Guarda el **orden y minuto exacto de cada compra**. |
| **Data Dragon** | Campeones, objetos, runas y hechizos del parche actual; iconos desde el CDN de Riot. |
| **Dataset** | Picks, bans, lado, ganador, KDA, oro, visión, inventarios ordenados y estadísticas de equipo. |
| **DraftNet** | Set-transformer sobre 10 picks + 10 bans. Soporta drafts incompletos (slots desconocidos). |
| **BuildNet** | Contexto (campeón, rol, aliados, rivales) + GRU sobre el historial de compras → siguiente objeto. |
| **Motor de recomendación** | Mezcla modelo + winrates suavizados + matchups + sinergias + presencia; para objetos añade slot, precio, componentes y perfil de daño rival. |
| **API** | FastAPI con endpoints de draft, build, análisis en vivo, meta, dataset y modelo. |
| **Interfaz** | SPA HTML/CSS/JS sin frameworks, tema hextech, responsive, accesible y con gráficos SVG propios. |
| **Cliente local (LCU)** | Lee champ select y partida en curso desde League of Legends. Busca la instalación en todos los discos y sincroniza Draft/En vivo; solo lectura. |
| **Escritorio** | `RiftSense.exe` con Chromium embebido (Electron): arranca el backend, abre la UI y lo detiene al cerrar. |

---

## 2. Puesta en marcha rápida

Requisitos: **Python 3.11+** (probado en 3.14) y conexión a internet para el
scraping. Node no es necesario.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1) Catálogo estático (data/static + artifacts/meta.json)
.\.venv\Scripts\python.exe -m lolanalyzer.scraper.ddragon

# 2) Partidas profesionales (≈41.000 partidas, 5-10 min)
.\.venv\Scripts\python.exe -m lolanalyzer.scraper.leaguepedia --since 2024-01-01

# 3) Dataset + priors
.\.venv\Scripts\python.exe -m lolanalyzer.data.build_dataset

# 4) Entrenar las dos redes
.\.venv\Scripts\python.exe -m lolanalyzer.models.train_draft
.\.venv\Scripts\python.exe -m lolanalyzer.models.train_build

# 5) Servir la app
.\.venv\Scripts\python.exe -m lolanalyzer.api.server
```

Abre **http://127.0.0.1:8000**.

Si League of Legends está abierto en este PC, la pestaña **Cliente** lo detecta
y puedes sincronizar champ select y partida sin clave adicional.

También puedes abrir la aplicación en una ventana de escritorio con Chromium
embebido (requiere compilar el lanzador una vez, ver sección 13).

También puedes ejecutar todo de una vez:

```powershell
.\.venv\Scripts\python.exe scripts\build_all.py
```

o usar `.\start.ps1` para crear el entorno, instalar dependencias y arrancar el
servidor (la primera vez hay que generar datos/modelos con el pipeline).

---

## 3. Dataset generado

Estado de la ejecución incluida en este repositorio:

| Métrica | Valor |
| --- | --- |
| Partidas descargadas | **41.124** |
| Filas de jugador descargadas | **411.240** |
| Partidas válidas (10 jugadores, roles y ganador) | **40.805** |
| Builds de jugador válidas | **407.797** |
| Rango temporal | 2024-01-01 → 2026-11-15 |
| Campeones con datos | **172 / 173** |
| Cobertura de objetos | **99,6%** |
| Splits cronológicos | 34.685 train · 4.080 val · 2.040 test |

Fuente: [Leaguepedia](https://lol.fandom.com/) (API Cargo) + [Riot Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon).
El dataset completo de Leaguepedia supera **1,38 millones** de filas de jugador;
aquí se filtran los parches recientes para que el meta sea útil.

### Cómo se construye

1. **Descarga**: `ScoreboardGames` y `ScoreboardPlayers` desde `Special:CargoExport`.
   Checkpoint en `data/raw/leaguepedia/checkpoint.json`; si se corta, reanuda.
2. **Normalización**: los nombres de campeones/objetos se cruzan con Data Dragon
   con alias y fuzzy matching. Se priorizan las variantes de la Grieta sobre las
   de Arena (mismo nombre, distinto id).
3. **Validación**: se exigen 10 jugadores, 5 roles por equipo y ganador conocido.
4. **Builds**: los inventarios se guardan en orden de compra (según el orden del
   campo `Items`). Con Riot timelines se añade el minuto exacto.
5. **Splits**: cronológicos, nunca aleatorios. El test es el 5% más reciente.
6. **Priors**: winrate/presencia/sinergias/matchups suavizados con solo el split
   de entrenamiento.

---

## 4. Modelos

### DraftNet

- Entrada: 10 picks + 10 bans como tokens `(campeón, rol, lado, tipo)`.
- `nn.TransformerEncoder` de 2 capas, `d_model=64`, 4 cabezas, pre-norm.
- Pooling separado por equipo, features `[A, B, A−B, A·B]` y cabeza MLP.
- **Drafts parciales**: durante el entrenamiento se ocultan picks (10%) usando un
  embedding `UNK`; los slots desconocidos conservan rol y lado. Así la UI puede
  recomendar con solo 2-3 picks conocidos.
- Pesos por recencia (vida media ≈ 1 año) y `BCEWithLogitsLoss`. Dropout 0,20 y weight decay para frenar el overfitting.
- Salida: `P(gana el equipo 1 / lado azul)`. Para el lado rojo se invierte.

### BuildNet

- Contexto: campeón, rol, lado, media de aliados, media de rivales, slot actual.
- Secuencia: embeddings de los objetos ya comprados → GRU.
- Cabeza MLP → softmax sobre los 205 objetos comprables del vocabulario.
- Entrenamiento con un prefijo aleatorio por build y epoch; pérdida CE con label
  smoothing. Filtro de parche mínimo (`BUILD_MIN_PATCH=25.10`).
- Salida: distribución del **siguiente** objeto. En inferencia se filtra por
  inventario, botas duplicadas, precio y componentes asequibles.

### Motor de decisión

Para campeones:

```
score = z(modelo) + z(winrate rol) + z(matchup) + z(sinergia) + z(presencia)
        (0.45)      (0.25)         (0.15)      (0.10)        (0.05)
```

Los pesos se renormalizan si falta información (p. ej., sin rivales no hay
matchup). El score 0-100 es **relativo a los candidatos**, no una probabilidad.
La probabilidad de victoria se muestra aparte y lleva su aviso.

Para objetos:

```
score = z(modelo) + z(frecuencia) + z(slot) + z(winrate) + z(ajuste defensivo)
        (0.50)      (0.20)       (0.15)    (0.05)         (0.10)
```

Además: componentes de la ruta de build, oro restante, perfil AD/AP del rival y
winrate empírico por número de objetos (asociación, no causalidad).

### Métricas

Las métricas finales están en `artifacts/draft_report.json` y
`artifacts/build_report.json` y se muestran en la pestaña **Modelo**. Incluyen
AUC, accuracy, log-loss, Brier, calibración, Top-1/3/5 y MRR, además de una
evaluación específica con 2 y 4 picks desconocidos.

> En esta ejecución DraftNet obtiene **AUC 0,540 y accuracy 53,9% en test**
> (0,556 en validación). El draft explica una parte pequeña del resultado: la
> herramienta ayuda a ordenar opciones, no a predecir el futuro. BuildNet ronda
> **Top-3 ≈ 50%** al predecir el siguiente objeto, muy por encima del baseline
> de popularidad.

---

## 5. High elo con Riot API (opcional)

El dataset pro no necesita clave. Para añadir soloQ de alto elo y tiempos
exactos de compra:

1. Copia `.env.example` a `.env` y rellena `RIOT_API_KEY`.
2. Descarga apex leagues:

```powershell
.\.venv\Scripts\python.exe -m lolanalyzer.scraper.riot_api --region euw1 --players 25 --matches 20
```

3. Reconstruye el dataset mezclando fuentes:

```powershell
.\.venv\Scripts\python.exe -m lolanalyzer.data.build_dataset --sources leaguepedia,riot
```

El adaptador:

- Recorre Challenger, Grandmaster y Master y ordena por LP.
- Deduplica partidas, cachea respuestas en `data/cache/riot/`.
- Filtra `queue=420`, Grieta del Invocador y duración ≥ 10 min.
- Convierte todo al esquema de Leaguepedia (mismos JSONL) y añade
  `PurchaseTimes` con el minuto de cada compra.
- Maneja 429 con `Retry-After` y avisa si la clave caducó (403).

Las claves de desarrollo de Riot caducan cada 24 h y tienen límites estrictos.
No metas la clave en el código ni la subas al repositorio.

---

## 6. Cliente local (LCU)

RiftSense puede leer tu League of Legends **sin clave de Riot** y sin tocar el
juego. Solo lectura: no automatiza picks, bans ni compras.

### Qué lee

| Momento | Fuente | Datos |
| --- | --- | --- |
| Cliente abierto | `gameflow-phase` + `current-summoner` | fase, invocador, nivel |
| Champ select | `lol-champ-select/v1/session` | picks, bans, rol, lado, timer |
| Partida | `https://127.0.0.1:2999/liveclientdata/allgamedata` | campeón, objetos, oro, minuto, KDA, CS, equipos |

### Cómo usarlo

1. Abre League of Legends en el mismo PC donde corre RiftSense.
2. Entra en **Cliente**. Debe aparecer `LCU conectado` y tu invocador.
3. En champ select pulsa **Usar este draft** o activa
   **Sincronizar automáticamente**. El slot de tu rol se deja libre para comparar
   alternativas; tu campeón hover se muestra en el panel.
4. En partida pulsa **Analizar en vivo** o deja el auto activo. La pestaña
   **En vivo** se rellena con campeón, inventario, oro y minuto reales.
5. En **Draft** tienes **Importar cliente**; en **En vivo**, **Leer cliente**
   y **Auto**.

La sincronización consulta cada 3 s y se pausa con la pestaña oculta.

### Búsqueda automática

Al arrancar el servidor, RiftSense busca League of Legends en segundo plano y
muestra en **Cliente** el disco y la ruta encontrados. Fuentes, en orden:

1. `LOL_LOCKFILE` / `LOL_INSTALL_DIR`.
2. Ruta guardada en `data/client_location.json`.
3. Metadata del Riot Client (`product_install_full_path`).
4. `RiotClientInstalls.json`.
5. Registro de Windows (entrada de desinstalación de LoL).
6. Rutas habituales en **todas** las unidades (C:, D:, E:...).
7. Escaneo de todas las unidades fijas/extraíbles (máx. 45 s) si no aparece.

La búsqueda no exige que LoL esté abierto: encuentra la instalación aunque el
cliente esté cerrado. El lockfile solo existe con el cliente en ejecución.

En **Cliente** puedes:

- Ver el disco y la ruta (`D:\Riot Games\League of Legends`).
- Pulsar **Buscar de nuevo**.
- Escribir una ruta manual y pulsar **Usar ruta**; se guarda y se reutiliza.

La búsqueda corre en un hilo aparte y no bloquea la interfaz. Si tu instalación
está en una ruta rara, el escaneo de discos la encuentra o puedes fijarla a mano.

### Instalación personalizada

RiftSense busca el `lockfile` en las rutas habituales de Windows, el registro
y las unidades C-G. Si tu cliente está en otra ruta:

```env
LOL_LOCKFILE=D:\Riot Games\League of Legends\lockfile
```

Reinicia el servidor después de cambiar `.env`. El lockfile del **Riot Client**
se ignora: solo se acepta el proceso `LeagueClient`.

### Seguridad y límites

- El servidor escucha en `127.0.0.1`; el LCU es local.
- La contraseña efímera del lockfile se usa solo en memoria. No se muestra en
  la API ni en la interfaz.
- No se envían inputs al juego: no hay auto-pick, auto-ban ni auto-compra.
- El LCU **no es una API oficial** de Riot y puede cambiar. Si falla, la app
  sigue funcionando en modo manual.
- El Live Client Data API no está disponible en repeticiones/espectador.
- En ARAM o modos con campeones duplicados se deduplican nombres; si falta la
  posición, se infiere el rol más jugado del campeón en el dataset.

### Diagnóstico

| Síntoma | Causa / solución |
| --- | --- |
| `Cliente no detectado` | Abre LoL; el servidor debe correr en el mismo Windows. Pulsa **Buscar de nuevo** o fija la ruta con **Usar ruta**. |
| Detecta el Riot Client | Ya se ignora; define `LOL_LOCKFILE` si tu ruta es personalizada. |
| `Live API en espera` | Solo aparece dentro de una partida cargada (puerto 2999). |
| Champ select vacío | El LCU puede tardar 1-2 s; pulsa Actualizar. |
| Error 401/403 del LCU | Reinicia el cliente: el lockfile cambia en cada sesión. |

---

## 7. API

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/api/health` | Estado del motor y modelos. |
| GET | `/api/meta` | Campeones, objetos, stats y estado. |
| GET | `/api/meta/champions?role=MID` | Meta filtrada por rol. |
| POST | `/api/draft` | Recomendaciones de pick para un draft parcial. |
| POST | `/api/draft/bans` | Prioridad de ban. |
| POST | `/api/build` | Siguiente objeto + plan de build. |
| POST | `/api/live` | Draft + build en una sola llamada. |
| GET | `/api/dataset` | Resumen, cobertura y tabla del meta. |
| GET | `/api/model` | Reportes y artefactos. |
| POST | `/api/reload` | Recarga los modelos sin reiniciar. |
| GET | `/api/client/status` | Estado del LCU, fase, invocador y Live API. |
| GET | `/api/client/draft` | Picks, bans, rol y lado del champ select. |
| GET | `/api/client/live` | Campeón, objetos, oro y minuto de la partida. |
| POST | `/api/client/search` | Vuelve a buscar LoL en todos los discos. |
| POST | `/api/client/path` | Fija una ruta manual y la guarda. |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/api/draft \
  -H "Content-Type: application/json" \
  -d '{"role":"MID","side":"blue","bans":["Yone","Aurora"],
       "allies":["K'Sante","Lee Sin",null,"Jinx","Rakan"],
       "enemies":["Gnar","Vi","Orianna","Kai'Sa","Nautilus"]}'
```

Swagger interactivo en **http://127.0.0.1:8000/docs**.

---

## 8. Interfaz

- **Draft**: elige lado y rol, rellena aliados/rivales/bans y obtén los mejores
  picks con evidencia desglosada (modelo, winrate, matchup, sinergia, presencia).
  Incluye prioridad de ban y un draft de ejemplo.
- **En vivo**: campeón, rol, oro, minuto, composiciones e inventario. Devuelve la
  siguiente compra, componentes asequibles, plan completo, perfil AD/AP, spikes
  por número de objetos y una estimación de victoria del draft.
- **Cliente**: estado del LCU, ubicación y búsqueda en todos los discos, champ
  select (picks/bans/rol) y partida en curso (campeón, objetos, oro, minuto).
  Botones de importación y auto-sync.
- **Modelo**: métricas, curvas de entrenamiento, calibración y arquitectura.
- **Datos**: resumen del dataset, cobertura, torneos, pipeline y tabla del meta
  con búsqueda, filtro por rol y ordenación.

Diseño responsive de 320 px a 4K, navegación por teclado, diálogos nativos,
`prefers-reduced-motion` y estados de carga/error. Los iconos vienen del CDN de
Riot con fallback a iniciales si no hay red.

---

## 9. Estructura

```
.
├── lolanalyzer/
│   ├── config.py                 rutas y constantes
│   ├── utils.py                  HTTP, JSON, normalización
│   ├── scraper/
│   │   ├── ddragon.py            catálogo estático
│   │   ├── leaguepedia.py        partidas pro (CargoExport)
│   │   └── riot_api.py           high elo + timelines
│   ├── data/
│   │   └── build_dataset.py      dataset, vocab, priors, timings
│   ├── models/
│   │   ├── draft_net.py          arquitectura DraftNet
│   │   ├── build_net.py          arquitectura BuildNet
│   │   ├── train_draft.py
│   │   └── train_build.py
│   ├── api/
│   │   ├── engine.py             inferencia + recomendación explicada
│   │   ├── schemas.py
│   │   └── server.py             FastAPI + estáticos
│   └── lcu/
│       ├── client.py             conector LCU + Live Client API
│       ├── locator.py            búsqueda en discos + lockfile
│       └── mapper.py             champ select y partida -> esquema RiftSense
├── web/                          SPA (HTML/CSS/JS, sin build)
├── desktop/                      lanzador Electron (Chromium embebido)
│   ├── main.js                   arranque del servidor + ventana
│   ├── splash.html               pantalla de carga
│   ├── package.json
│   └── scripts/                  icono y raíz del proyecto
├── data/raw, data/processed      datos (no versionados)
├── artifacts/                    meta.json, modelos y reportes
├── scripts/build_all.py          pipeline completo
├── start.ps1
└── requirements.txt
```

---

## 10. Reentrenar con un parche nuevo

```powershell
# Vuelve a descargar el rango completo (los parches cambian el meta)
.\.venv\Scripts\python.exe -m lolanalyzer.scraper.leaguepedia --since 2024-01-01 --restart
.\.venv\Scripts\python.exe -m lolanalyzer.scraper.ddragon --force
.\.venv\Scripts\python.exe -m lolanalyzer.data.build_dataset
.\.venv\Scripts\python.exe -m lolanalyzer.models.train_draft
.\.venv\Scripts\python.exe -m lolanalyzer.models.train_build
```

Con el servidor en marcha, `POST /api/reload` recarga los checkpoints.

---

## 11. Límites y transparencia

- **Datos profesionales ≠ tu elo.** El comportamiento en soloQ puede diferir.
  Usa el adaptador de Riot para acercarte a tu contexto.
- **Sin integración con el cliente de LoL.** No lee la partida en curso: tú
  introduces campeones, oro, minuto e inventario. La Riot API no permite leer
  partidas en vivo sin herramientas externas.
- **Orden de compra aproximado** en Leaguepedia. Los timelines de Riot lo hacen
  exacto, pero no se usan en el modelo actual (sí se guardan para análisis y
  timings).
- **Winrate por número de objetos** es correlación; ganar te da oro, no al revés.
- **Draft** explica poco del resultado. Los porcentajes son estimaciones
  calibradas, no certezas.
- **Cobertura**: algunos objetos nuevos/evento no existen aún en Data Dragon; se
  descartan y se reporta la cobertura.
- Proyecto no afiliado a Riot Games. League of Legends es una marca de Riot
  Games, Inc.

---

## 12. Problemas frecuentes

| Problema | Solución |
| --- | --- |
| `Cargo error: ratelimited` | El scraper reintenta con backoff. Espera o reduce el rango con `--since`. |
| `403` de Riot | Clave caducada (24 h). Genera otra en el portal. |
| `Faltan datos procesados` | Ejecuta `build_dataset` antes que el servidor. |
| `torch` no instala | Usa Python 3.11-3.14 y `pip install torch` desde PyPI. |
| Puerto ocupado | `python -m lolanalyzer.api.server --port 8010`. |
| Iconos no cargan | Sin internet el CDN falla; la UI muestra iniciales. |

---

## 13. Aplicación de escritorio (.exe)

`RiftSense.exe` es un lanzador con **Chromium embebido (Electron)**. Arranca el
servidor Python si hace falta, muestra una pantalla de carga y abre la interfaz
en una ventana nativa.

### Uso

1. Deja `RiftSense.exe` en la raíz del proyecto, junto a `.venv`, `lolanalyzer`
   y `web`.
2. Doble clic. Si el servidor ya está corriendo lo reutiliza; si no, lo arranca
   en `127.0.0.1:8000` (o en 8765+ si el puerto está ocupado).
3. Al cerrar la ventana, si el lanzador arrancó el servidor, lo apaga.

No es un ejecutable autónomo: necesita el proyecto y `.venv`. Para distribuirlo
a otros PC habría que empaquetar Python, PyTorch y los artefactos.

### Compilar

```powershell
cd desktop
npm install
npm run dist
Copy-Item .\dist\RiftSense.exe ..\RiftSense.exe
```

- Requiere Node 20+ y conexión para descargar Electron (una vez).
- El `.exe` pesa ~106 MB.
- No está firmado: Windows SmartScreen puede pedir confirmación.
- El icono se genera desde `web/assets/favicon.svg`.
- `root.json` se genera con la ruta del proyecto y se incrusta en el `.exe`.

Si mueves el proyecto, vuelve a compilar o crea `rift-sense.config.json` junto
al `.exe`:

```json
{ "root": "C:\\ruta\\al\\proyecto" }
```

### Logs

- `%APPDATA%\riftsense-desktop\server.log`
- `%APPDATA%\riftsense-desktop\launcher.log`
- `%TEMP%\riftsense-boot.log`

### Nota sobre CEF

Se usa Electron, que embebe Chromium. No es el proyecto CEF/CefSharp; si
necesitas una integración CEF nativa, el mismo flujo se puede portar a
CefSharp/.NET.

