# Travel Guide RAG

## Autor

Nombre: Raúl Sánchez Serrano

## Repositorio

https://github.com/RaulRX/travel-guide-RAG

## Contexto del proyecto

Este proyecto consiste en el desarrollo de un asistente turístico conversacional sobre la isla de Tenerife, construido en torno a un LLM comercial accedido vía API.

El asistente integra tres capacidades principales en un único notebook reproducible:

- **RAG (Retrieval-Augmented Generation)**: recuperación semántica de información sobre una guía turística (chunking, embeddings, vector store y respuestas con cita de la fuente).
- **Diálogo multiturno**: gestión de un historial de conversación que mantiene el contexto entre turnos, controlando su longitud para no exceder el límite de tokens del modelo.
- **Function calling**: al menos una función externa (`get_weather`) invocable por el modelo, con gestión básica de errores y registro de las llamadas.

El objetivo general es desarrollar un prototipo conversacional reproducible que combine búsqueda semántica, mantenimiento de contexto y llamadas a servicios externos, todo desde un único notebook bien documentado.

## Estructura del proyecto

```
travel-guide-RAG/
├── .env                          # Variables de entorno necesarias para ejecutar el notebook production-rag (API keys, parámetros del modelo, etc.)
├── .env-template                 # Plantilla de referencia para crear el archivo .env
├── Makefile                      # Automatiza la creación del entorno (uv), registro del kernel de Jupyter y arranque de JupyterLab
├── pyproject.toml                # Definición del proyecto y dependencias (gestionado con uv)
├── requirements.txt              # Dependencias del proyecto en formato pip
├── uv.lock                       # Lockfile de dependencias generado por uv
├── README.md                     # Este documento
├── data/
│   ├── Enunciado Entrega Final.pdf  # Guía de la entrega final proporcionada por el profesorado
│   ├── TENERIFE.pdf                  # Guía turística de Tenerife usada como base de conocimiento del RAG
│   └── templates/
│       └── system_instructions.txt   # Plantilla del system prompt del asistente turístico
└── notebook/
    ├── final/
    │   └── production-rag.ipynb      # Notebook principal con la solución completa (RAG, diálogo multiturno y function calling)
    └── tests/
        └── testing-rag.ipynb          # Notebook de pruebas y experimentación previa a la versión final
```

## Diseño de la solución

- **Modelo de lenguaje**: `gemini-3.1-flash-lite` (Google Gemini), accedido a través del SDK `google-genai`.

- **Modelo de embeddings**: `gemini-embedding-001`, usado para vectorizar tanto los chunks de la guía como las consultas del usuario en el *similarity search*. El documento (`TENERIFE.pdf`) se divide en fragmentos de `CHUNK_SIZE = 850` caracteres con un `CHUNK_OVERLAP = 400`.

- **Vector store**: **FAISS** (`langchain_community.vectorstores.FAISS`), en memoria y sin servicios externos.

- **Splitter**: `RecursiveCharacterTextSplitter` (LangChain), con `add_start_index=True` para conservar la posición original de cada chunk dentro del documento y poder citarlo como fuente.

- **Configuración del modelo (`model_config`)**: generada mediante `generate_configuration()`, que construye un `GenerateContentConfig` con:
  - `system_instruction` cargada desde `data/templates/system_instructions.txt`, donde se indican las instrucciones de sistema al modelo para especializar al modelo y gestionar la respuestas asi como el uso de herramientas
  - `temperature`, `top_k`, `top_p` y `seed` parametrizables vía `.env` (valores por defecto: `temperature=0.2`, `top_k=13`, `top_p=0.8`, `seed=1111`) para la gestión y decisión de respuestas
  - `max_output_tokens` por pregunta (`MAX_OUTPUT_TOKENS_PER_QUESTION`).
  - `thinking_config` con `ThinkingLevel.MEDIUM`.
  - `tools` y `tool_config` con `FunctionCallingConfigMode.AUTO`, para que el modelo decida de forma autónoma cuándo invocar la herramienta disponible.

- **Tool de movilidad**: `get_city_mobility(origin, destination, transport_mode="best")`, una herramienta que **simula una API de transporte de Tenerife** (TITSA) con fines de prueba. A partir de un dataset embebido de pueblos y POIs reales de la isla, devuelve el modo de transporte recomendado, el tiempo estimado en minutos, el pueblo más cercano al origen y la línea de guagua/punto de recogida. Siempre responde con un `dict` serializable a JSON (`status="OK"` o `status="ERROR"` con un `error_code`), consumido directamente por el LLM vía *function calling*.

- **Uso conversacional**: el chatbot se ejecuta sobre una sesión de chat (`llm_client.chats.create`) que mantiene el historial de la conversación de forma nativa. En cada turno (`run_chat_conversation`):
  1. Se enriquece el mensaje del usuario con el contexto recuperado del vector store (RAG).
  2. Se envía al modelo dentro de la sesión de chat, que conserva el contexto multiturno.
  3. Si la respuesta incluye una `function_call`, se detecta y ejecuta la única tool soportada (`get_city_mobility`), y su resultado se reenvía al modelo como `FunctionResponse` para obtener la respuesta final.

- **Gestión de tokens de salida como mecanismo de control**: se parte de `MODEL_MAX_OUTPUT_TOKENS` (límite del modelo) y, tras cada turno, se calcula el consumo real de `output_tokens` se descuenta del presupuesto restante (`output_tokens_remain`). Al finalizar, se muestran estadísticas (`average_output_tokens`) con el promedio, mínimo y máximo de tokens consumidos por turno.

- **Programa principal**: la celda bajo `CHAT BOT MAIN` es el punto de entrada interactivo del notebook. Inicializa la sesión de chat con la configuración y la tool de movilidad, y lanza un bucle de conversación mediante `input()`: solicita la pregunta del usuario, ejecuta `run_chat_conversation`, pregunta si se desea continuar y, al terminar, muestra las estadísticas de tokens y el historial completo de la conversación.

## Decisiones técnicas

- **Modelo de lenguaje (`gemini-3.1-flash-lite`)**: se eligió un modelo *flash-lite* por su bajo coste y baja latencia, suficientes para un asistente conversacional de preguntas y respuestas sobre una guía turística.

- **Embeddings y chunking (`gemini-embedding-001`, `CHUNK_SIZE=850`, `CHUNK_OVERLAP=400`)**: se usó un solape superior al 20%-40% respecto al tamaño de chunk para preservar el contexto entre fragmentos consecutivos y minimizar la pérdida de información en los límites de cada chunk.

- **Vector store (FAISS)**: se priorizó la sencillez de la solución frente a alternativas gestionadas (Chroma, Pinecone, etc.), suficiente para el volumen de datos de una única guía turística y para un prototipo de notebook.

- **`system_instruction` en fichero externo (`data/templates/system_instructions.txt`)**: el comportamiento y la especialización del modelo como guía turístico se definen en un fichero de instrucciones independiente, sin acoplarlo a una zona geográfica concreta (Tenerife), de modo que el mismo prompt pueda reutilizarse para otros destinos en el futuro sin modificar el código.

- **`model_config` (parámetros de generación)**:
  - **`temperature=0.2`**: valor bajo para favorecer respuestas fieles al contexto recuperado (RAG) y minimizar alucinaciones, alineado con un asistente informativo más que creativo.
  - **`top_k=13` y `top_p=0.8`**: Con ambos parámetros quiero que el modelo
  elija los mejores tokens para una respuesta más acertada, pero puede ser un extra en la parametrización que no esté correctamente aplicada
  - **`seed=1111`**: fijado para favorecer la reproducibilidad de los resultados
  - **`max_output_tokens=350`**: valor maximo de tokens de respuesta. Esta ajustado según las comprobaciones realizadas en el notebook 'testing-rag' con las preguntas de evaluación. Aún habría que realizar mayor cantidad de pruebas.
  - **`response_logprobs=False`**: desactivado para evitar coste/latencia adicional en producción. Podría explorarse como señal de confianza en respuestas futuras.
  - **`thinking_config=ThinkingLevel.MEDIUM`**: La intención es que el modelo tenga un minimo de esfuerzo a la hora de elaborar las respuestas, pero puede conllevar a una latencia en un modelo que se ha elegido por su eficiencia, aunque puede ayudar a dar mejores respuestas.
  - **`tools` / `tool_config=AUTO`**: Modo de ejecución de las herramientas donde la decisión se deja al modelo para el diseño híbrido RAG + tool. Además, con las instrucciones del sistema se le indica al modelo que solo lo use en caso de que el usuario solicite información asociada al uso de la herramienta

- **Limpieza del documento (`load_cleanup_documents`)**: en `TENERIFE.pdf` se eliminan referencias a imágenes y enlaces, espacios y saltos de línea sobrantes, y páginas que quedan en blanco tras la limpieza. El objetivo es mantener la información lo más compacta posible en esta primera versión, reduciendo ruido en los chunks indexados.

## Limitaciones

- Calcular los tokens de salida turno a turno no es tan directo como parece: `usage_metadata` solo está disponible al final de la respuesta, así que para llevar la cuenta en cada llamada hay que recurrir al contador de tokens sobre la propia respuesta del modelo, no a un valor que se pueda consultar "sobre la marcha".

- Usar schemas con Pydantic directamente no es tan sencillo en Gemini: no se puede pasar el schema tal cual, hay que transformarlo a un `Tool` para poder incluirlo en la configuración del modelo.

- Al eliminar páginas en blanco, imágenes y enlaces durante la limpieza del PDF, se pierde en parte la referencia real de las páginas, así como las imágenes y enlaces que podrían dar al usuario una experiencia más completa.

## Mejora futura

- Añadir una capa agéntica para que el propio modelo gestione el flujo de llamadas a herramientas, sin tener que controlarlo manualmente.
- Definir los schemas de las tools con objetos Pydantic en lugar de diccionarios.
- Mejorar la forma en que se registran y se incluyen las herramientas en la configuración del modelo.
- Persistir los vectores de los documentos en una base de datos en lugar de regenerarlos en cada ejecución.
- Montar una evaluación de las preguntas de prueba organizada por categorías.
- Recuperar y mostrar enlaces e imágenes de los documentos para dar al usuario una mejor perspectiva de los lugares y una guía más completa de cómo llegar.
- Construir una interfaz de usuario amigable que permita navegar por el historial e incluso ver rutas o imágenes en un mapa.
- Añadir tools que llamen a Apis y no a simulaciones.-
