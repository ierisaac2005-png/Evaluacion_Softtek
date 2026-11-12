"""Actividad 4: evaluación del recomendador original, sin ajustar su algoritmo.

Relevancia temática binaria sobre todos los candidatos salvo la semilla.
Ejecutar: python evaluacion.py --base .
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recomendador import Recomendador

SEMILLA = 'DOC008'
NECESIDAD = ('Localizar políticas, procedimientos y experiencias sobre retención, '
             'integridad y restauración de respaldos después de un respaldo nocturno '
             'fallido por falta de espacio.')


def evaluar(base: str | Path = '.') -> dict:
    """Reproduce Top-5 y contrasta sus IDs con juicios externos al ranking."""
    base = Path(base)
    salida = base / 'evidencias'
    salida.mkdir(parents=True, exist_ok=True)
    modelo = Recomendador(base / 'corpus.csv')
    # Se carga el juicio de todos los candidatos, no solo de los recuperados.
    verdad = pd.read_csv(base / 'ground_truth.csv')
    candidatos = modelo.df.loc[modelo.df.id != SEMILLA, ['id', 'title']]
    if verdad.id.duplicated().any() or set(verdad.id) != set(candidatos.id):
        raise ValueError('El ground truth debe cubrir cada candidato exactamente una vez.')
    if not verdad.relevante.isin([0, 1]).all():
        raise ValueError('La etiqueta de relevancia debe ser 0 o 1.')
    top = modelo.recomendar(documento_id=SEMILLA, k=5)
    assert len(top) == 5 and SEMILLA not in set(top.id)
    validacion = candidatos.merge(verdad, on='id', validate='one_to_one')
    validacion['recuperado'] = validacion.id.isin(top.id).astype(int)
    # 1 significa relevante en y_true y recuperado en y_pred.
    y_true = validacion.relevante.to_numpy()
    y_pred = validacion.recuperado.to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    valores = [precision_score(y_true, y_pred, zero_division=0),
               recall_score(y_true, y_pred, zero_division=0),
               f1_score(y_true, y_pred, zero_division=0)]
    metricas = pd.DataFrame({'metrica': ['Precision', 'Recall', 'F1-score'], 'resultado': valores})
    validacion['resultado'] = [
        'VP' if t and r else 'FN' if t else 'FP' if r else 'VN'
        for t, r in zip(y_true, y_pred)]
    # Verificación independiente de las fórmulas; no son etiquetas predichas a mano.
    assert np.allclose(valores, [tp/(tp+fp), tp/(tp+fn), 2*tp/(2*tp+fp+fn)])
    assert tp+fp+fn+tn == len(candidatos)
    top.to_csv(salida / 'top5.csv', index=False)
    validacion.to_csv(salida / 'tabla_validacion.csv', index=False)
    metricas.to_csv(salida / 'metricas.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    ConfusionMatrixDisplay(cm, display_labels=['No relevante', 'Relevante']).plot(
        ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_xticklabels(['No recuperado', 'Recuperado'])
    ax.set_xlabel('Salida del sistema'); ax.set_ylabel('Ground truth')
    ax.set_title('Matriz de confusión | DOC008 | Top-5')
    fig.tight_layout(); fig.savefig(salida / 'matriz_confusion.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(metricas.metrica, valores, color=['#175c80', '#187e83', '#385773'], width=.55)
    ax.bar_label(barras, labels=[f'{v:.2f}' for v in valores], padding=5)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Valor (0 a 1)')
    ax.set_title('Calidad de recuperación | mismo caso y mismo K')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(salida / 'metricas.png', dpi=180); plt.close(fig)
    # El ranking ampliado es solo diagnóstico: el sistema evaluado sigue en K=5.
    diagnostico = modelo.recomendar(documento_id=SEMILLA, k=29)
    omitidos = diagnostico.loc[diagnostico.id.isin(validacion.query('resultado == "FN"').id)]
    registro = dict(semilla=SEMILLA, necesidad=NECESIDAD, k=5,
                    documentos=len(modelo.df), candidatos=len(candidatos),
                    relevantes=int(y_true.sum()), recuperados=int(y_pred.sum()),
                    precision=valores[0], recall=valores[1], f1=valores[2],
                    VP=tp, FP=fp, FN=fn, VN=tn, matriz=cm.tolist(),
                    corpus_sha256=hashlib.sha256((base/'corpus.csv').read_bytes()).hexdigest(),
                    recomendador_sha256=hashlib.sha256((base/'recomendador.py').read_bytes()).hexdigest(),
                    versiones={'python':platform.python_version(),'sklearn':sklearn.__version__,
                               'pandas':pd.__version__,'numpy':np.__version__},
                    alcance='Evaluación temática de un caso simulado; no mide resolución de incidentes.')
    (salida / 'resultados.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
    texto = '\n'.join([
        'ACTIVIDAD 4 - EJECUCIÓN REAL DEL EVALUADOR',
        'Semilla: '+SEMILLA, 'Necesidad: '+NECESIDAD,
        'Corpus: 30 documentos. Universo evaluado: 29 (semilla excluida).',
        '\nTOP-5\n'+top.to_string(index=False, float_format=lambda v:f'{v:.6f}'),
        '\nENTRADAS Y CLASIFICACIÓN DE LOS 29 CANDIDATOS\n'+validacion.to_string(index=False),
        '\ny_true = '+str(y_true.tolist()), '\ny_pred = '+str(y_pred.tolist()),
        f'\nVP={tp}; FP={fp}; FN={fn}; VN={tn}',
        '\nMÉTRICAS\n'+metricas.to_string(index=False),
        '\nMatriz [[VN, FP], [FN, VP]] = '+str(cm.tolist()),
        '\nRELEVANTES OMITIDOS (diagnóstico)\n'+omitidos.to_string(index=False),
        '\nVersiones: '+str(registro['versiones']),
        'Validaciones de cobertura, exclusión de semilla y fórmulas: OK.',
        'EJECUCIÓN COMPLETADA SIN ERRORES.'])
    (salida/'ejecucion.txt').write_text(texto+'\n',encoding='utf-8')
    print(texto)
    return registro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=str(Path(__file__).resolve().parent))
    evaluar(parser.parse_args().base)
