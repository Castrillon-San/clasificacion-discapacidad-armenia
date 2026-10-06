# Clasificación de la categoría de discapacidad en Armenia (Quindío)

Proyecto integrador con metodología CRISP-DM. Aplicación web en Streamlit que estima la categoría de discapacidad más probable (física, intelectual, psicosocial, sensorial o múltiple) de una persona certificada en Armenia, a partir de su perfil sociodemográfico y de aseguramiento en salud.

La aplicación es una herramienta de apoyo a la planeación poblacional. No sustituye la valoración del equipo multidisciplinario de certificación ni debe usarse para decisiones sobre personas concretas.

## Datos

- **Fuente:** "Personas con discapacidad certificadas en Armenia", portal Datos Abiertos Colombia (datos.gov.co).
- **Periodo:** vigencias 2021 a 2025 (3.181 registros; 3.178 después de la limpieza).
- **Variable objetivo:** `categoria_discapacidad`, con visual, auditiva y sordoceguera agrupadas en la categoría sensorial.
- **Predictoras:** `edad`, `vigencia`, `genero`, `comuna`, `eps`, `regimen` y `condicion_victima`.

## Modelo

El modelo desplegado es una votación suave que combina regresión logística, KNN, árbol de decisión, SVM y Naive Bayes, con hiperparámetros ajustados por búsqueda en rejilla. El *pipeline* serializado incluye la imputación, la estandarización, la codificación *one-hot* y el balanceo por sobremuestreo, que solo actúa durante el entrenamiento.

| Métrica (conjunto de prueba) | Valor |
|---|---|
| F1 macro | 0,327 |
| *Accuracy* | 0,341 |
| *Precision* macro | 0,330 |
| *Recall* macro | 0,367 |
| F1 macro de la referencia aleatoria | 0,207 |

## Estructura del repositorio

```
├── app.py                      # Aplicación Streamlit
├── modelo_discapacidad.joblib  # Pipeline entrenado (generado en Colab)
├── metadatos_modelo.json       # Opciones válidas, métricas y versiones (generado en Colab)
├── requirements.txt            # Dependencias con las versiones del entrenamiento
├── plantilla_lote.csv          # Ejemplo de archivo para la predicción por lote
├── .streamlit/config.toml      # Tema visual de la aplicación
└── README.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones

- Las variables disponibles no incluyen información clínica, que es la que determina la categoría de discapacidad; por ello, el desempeño absoluto es moderado.
- El conjunto incluye solo personas certificadas.
- La discapacidad múltiple tiene un *recall* bajo (0,163).
- El tipo de documento se excluyó por redundancia con la edad; con ello, el modelo no distingue a la población migrante.
- Las predicciones para vigencias posteriores a 2025 suponen que la tendencia temporal observada se mantiene.
