# Actividad 4 · Evaluación del recomendador Softtek

**Isaac Esquivel Ruiz · Universidad Tecmilenio**  
Sistemas y tecnologías de administración del conocimiento

## Objetivo
Evaluar la recuperación Top-5 de la actividad 3 mediante precisión, recall, F1 y matriz de confusión, sin modificar el recomendador. Los 30 documentos son simulados; no son documentos internos de Softtek.

## Archivos
- `Actividad_4_Evaluacion_Softtek.ipynb`: notebook autónomo con código, salidas reales e interpretaciones.
- `evaluacion.py`: evaluación documentada con scikit-learn.
- `recomendador.py`: implementación original de la actividad 3, sin cambios.
- `corpus.csv`: mismo contenido de los 30 documentos de la actividad 3.
- `ground_truth.csv`: juicio binario y justificación de cada uno de los 29 candidatos.
- `Reporte_Actividad_4_Softtek.pdf`: arquitectura, escenario, tabla completa, métricas, análisis e infografía.
- `Infografia_Evaluacion_Softtek.png`: infografía independiente.
- `requirements.txt`: dependencias; versiones de la ejecución en `evidencias/resultados.json`.
- `evidencias/`: logs, datos y gráficas generadas.

## Ejecución en Colab
1. Sube `Actividad_4_Evaluacion_Softtek.ipynb` a Colab.
2. Selecciona **Entorno de ejecución → Ejecutar todas**.
3. El notebook crea `actividad4_resultados/` con el corpus, los scripts, el ground truth y `evidencias/`. No requiere subir el CSV ni conectar Drive.
4. Comprueba la salida “EJECUCIÓN COMPLETADA SIN ERRORES”.
5. Para documentar tu propia ejecución, toma una captura de la celda con las métricas y guárdala como `evidencias/captura_colab.png`.

El notebook entregado fue ejecutado localmente mediante IPython en proceso y conserva las salidas producidas. No se afirma que haya sido ejecutado en Colab. `salida_ejecucion.png` es una visualización del log real, no una captura de una interfaz. Se incluyen además el registro TXT y los gráficos originales.

## Ejecución local
Requiere Python 3.10 o posterior. Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python evaluacion.py --base .
```

El notebook contiene copias del código y los datos para ser autónomo. Si se editan archivos externos, esas copias no se actualizan automáticamente. Esta entrega evalúa los originales sin modificaciones.

## Escenario y ground truth
Semilla DOC008: respaldo nocturno fallido por falta de espacio. Se busca información sobre retención, integridad y restauración de respaldos.
Relevantes: DOC007, DOC009, DOC010, DOC011 y DOC012. Este criterio temático se conserva de la actividad 3 y se justifica revisando el contenido, sin usar el score para decidir relevancia.

Se evalúan los 29 candidatos. La semilla se excluye tanto del ranking como del universo de evaluación. Los documentos no recuperados se consideran negativos en `y_pred`, pero pueden seguir siendo relevantes en `y_true`.

## Resultados
| Medida | Valor |
|---|---:|
| Precision | 0.80 |
| Recall | 0.80 |
| F1-score | 0.80 |
| VP / FP / FN / VN | 4 / 1 / 1 / 23 |

Matriz con filas reales [no relevante, relevante] y columnas [no recuperado, recuperado]: `[[23, 1], [1, 4]]`.
DOC002 es el falso positivo. DOC009 es el falso negativo y aparece sexto en un ranking de diagnóstico; las métricas siguen usando exclusivamente K=5.

## Interpretación y límites
La precisión y el recall son iguales porque hay cinco relevantes y cinco recuperados. Se cubre el 80% de la información relevante según el criterio temático, pero falta un procedimiento de restauración. Un solo caso sintético no valida la calidad general ni mide ahorro de tiempo o resolución real.
Las mejoras (filtros, vocabulario, K, búsqueda híbrida y metadatos) se proponen para trabajo futuro; no se aplican en esta actividad.

## Evidencias
- `top5.csv`: ranking con scores.
- `tabla_validacion.csv`: los 29 candidatos, etiquetas, justificaciones y tipo de resultado.
- `metricas.csv`: métricas calculadas.
- `resultados.json`: conteos, versiones y hashes.
- `ejecucion.txt`: log completo con y_true, y_pred y verificaciones.
- `salida_ejecucion.png`: visualización del extracto del log.
- `matriz_confusion.png` y `metricas.png`: figuras originales.

