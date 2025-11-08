"""
================================================================================
APLICACIÓN STREAMLIT - SEGMENTACIÓN PANÓPTICA DE ÁRBOLES
================================================================================
Interfaz web para procesar imágenes usando el modelo Detectron2 entrenado.

Características:
- Procesamiento de imagen única
- Procesamiento por lotes
- Visualización interactiva de resultados
- Descarga de predicciones en formato JSON

Uso:
    streamlit run app_streamlit.py
================================================================================
"""

import streamlit as st
from pathlib import Path
import json
import tempfile
import zipfile
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
import requests
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Árboles - Deep Learning",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulo de predicción
try:
    from predictor import TreePanopticPredictor
    DETECTRON2_AVAILABLE = True
except ImportError as e:
    DETECTRON2_AVAILABLE = False
    st.error(f"⚠️ Error al importar módulos: {e}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_PATH = Path(r'model_final.pth')
OUTPUT_DIR = Path('outputs_streamlit')

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def load_model():
    """Carga el modelo (se cachea para no recargar en cada interacción)."""
    if not MODEL_PATH.exists():
        MODEL_URL = "https://huggingface.co/ua-darc/ua_deep_learning_model/resolve/main/model_final.pth"
        with st.spinner("Descargando el modelo... esto puede tomar un momento."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    try:
        with st.spinner("Cargando modelo Detectron2..."):
            predictor = TreePanopticPredictor(model_path=MODEL_PATH)

        st.success("✓ Modelo cargado correctamente")

        return predictor
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def create_visualization(image, instances, metadata, file_name, coverage):
    """
    Crea visualización de 4 paneles.

    Returns:
        fig: Figura de matplotlib
    """
    img_original = image.copy()

    # Panel 1: Original

    # Panel 2: Máscaras coloreadas por instancia
    img_masks = np.zeros_like(img_original)
    if len(instances) > 0:
        for i in range(len(instances)):
            mask = instances.pred_masks[i].numpy()
            color = plt.cm.rainbow(i / max(len(instances), 1))[:3]
            color = tuple(int(c * 255) for c in color)

            for c in range(3):
                img_masks[:, :, c] = np.where(mask, color[c], img_masks[:, :, c])

    # Panel 3: Visualización Detectron2
    
    v = Visualizer(
        img_original[:, :, ::-1],
        metadata=metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE
    )
    v = v.draw_instance_predictions(instances)
    img_detectron = v.get_image()[:, :, ::-1]

    # Panel 4: Overlay semitransparente
    img_overlay = img_original.copy()
    if len(instances) > 0:
        combined_mask = np.zeros_like(img_original, dtype=np.float32)
        for i in range(len(instances)):
            mask = instances.pred_masks[i].numpy()
            color = plt.cm.rainbow(i / max(len(instances), 1))[:3]
            color = np.array([c * 255 for c in color])

            for c in range(3):
                combined_mask[:, :, c] += mask.astype(float) * color[c]

        mask_any = instances.pred_masks.any(dim=0).numpy()
        img_overlay = img_overlay.astype(np.float32)
        img_overlay[mask_any] = img_overlay[mask_any] * 0.5 + combined_mask[mask_any] * 0.5
        img_overlay = img_overlay.astype(np.uint8)

    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original\n{file_name}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(img_masks, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Máscaras de Instancias\nN={len(instances)}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(img_detectron, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Detectron2 (Boxes + Máscaras)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Overlay Semitransparente\nCobertura: {coverage:.1f}%', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    return fig


def process_single_image(predictor, uploaded_file):
    """Procesa una imagen única."""

    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        # Predicción
        with st.spinner("Procesando imagen..."):
            result = predictor.predict(tmp_path)

        # Mostrar resultados
        st.success("✓ Predicción completada")

        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Instancias Detectadas", len(result['annotations']))
        with col2:
            individual = sum(1 for a in result['annotations'] if a['class'] == 'individual_tree')
            st.metric("Árboles Individuales", individual)
        with col3:
            groups = sum(1 for a in result['annotations'] if a['class'] == 'group_of_trees')
            st.metric("Grupos de Árboles", groups)

        # Cobertura
        st.metric("Cobertura de Dosel", f"{result['coverage']:.2f}%")

        # Visualización
        st.subheader("Visualización")
        fig = create_visualization(
            result['image'],
            result['instances'],
            predictor.metadata,
            uploaded_file.name,
            result['coverage']
        )
        st.pyplot(fig)

        # Guardar figura
        OUTPUT_DIR.mkdir(exist_ok=True)  # Verificar que el directorio existe
        output_path = OUTPUT_DIR / f"result_{uploaded_file.name.replace('.tif', '.png')}"
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Descargar resultados
        st.subheader("Descargar Resultados")

        col1, col2 = st.columns(2)

        with col1:
            # JSON con anotaciones
            json_str = json.dumps({
                'file_name': uploaded_file.name,
                'annotations': result['annotations'],
                'coverage': result['coverage']
            }, indent=2)

            st.download_button(
                label="📥 Descargar JSON",
                data=json_str,
                file_name=f"result_{uploaded_file.name.replace('.tif', '.json')}",
                mime="application/json"
            )

        with col2:
            # Imagen de visualización
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Descargar Visualización",
                    data=f,
                    file_name=output_path.name,
                    mime="image/png"
                )

        # Detalles de detecciones
        with st.expander("Ver detalles de detecciones"):
            for i, ann in enumerate(result['annotations'], 1):
                st.write(f"**Detección {i}:**")
                st.write(f"  - Clase: `{ann['class']}`")
                st.write(f"  - Confianza: `{ann['confidence_score']}`")
                st.write("---")

    finally:
        # Limpiar archivo temporal
        tmp_path.unlink()


def process_batch_images(predictor, uploaded_files):
    """Procesa múltiples imágenes."""

    st.info(f"Procesando {len(uploaded_files)} imágenes...")

    # Directorio temporal
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Guardar archivos
        for uploaded_file in uploaded_files:
            file_path = tmp_dir_path / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

        # Procesar por lotes
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = {}
        all_annotations = []

        for i, file_path in enumerate(sorted(tmp_dir_path.glob('*'))):
            status_text.text(f"Procesando: {file_path.name} ({i+1}/{len(uploaded_files)})")

            try:
                result = predictor.predict(file_path)
                results[file_path.name] = result
                all_annotations.extend(result['annotations'])
            except Exception as e:
                st.warning(f"⚠️ Error en {file_path.name}: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("✓ Procesamiento completado")

        # Guardar resultados en session_state para persistencia
        st.session_state.batch_results = results


def display_batch_results(predictor):
    """Muestra los resultados del procesamiento por lotes desde session_state."""

    if 'batch_results' not in st.session_state or not st.session_state.batch_results:
        return

    results = st.session_state.batch_results

    # Estadísticas generales
    st.success(f"✓ {len(results)} imágenes procesadas correctamente")

    st.subheader("Estadísticas Generales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_instances = sum(len(r['annotations']) for r in results.values())
        st.metric("Total Instancias", total_instances)

    with col2:
        total_individual = sum(
            sum(1 for a in r['annotations'] if a['class'] == 'individual_tree')
            for r in results.values()
        )
        st.metric("Árboles Individuales", total_individual)

    with col3:
        total_groups = sum(
            sum(1 for a in r['annotations'] if a['class'] == 'group_of_trees')
            for r in results.values()
        )
        st.metric("Grupos de Árboles", total_groups)

    with col4:
        avg_coverage = sum(r['coverage'] for r in results.values()) / len(results)
        st.metric("Cobertura Promedio", f"{avg_coverage:.2f}%")

    # Tabla de resultados
    st.subheader("Resultados por Imagen")

    import pandas as pd

    df_data = []
    for file_name, result in results.items():
        individual = sum(1 for a in result['annotations'] if a['class'] == 'individual_tree')
        groups = sum(1 for a in result['annotations'] if a['class'] == 'group_of_trees')

        df_data.append({
            'Archivo': file_name,
            'Total': len(result['annotations']),
            'Individuales': individual,
            'Grupos': groups,
            'Cobertura (%)': round(result['coverage'], 2)
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, width='stretch')

    # Visualizaciones de imágenes procesadas
    st.subheader("📸 Visualizaciones")

    # Opciones de visualización
    show_all = st.checkbox("Mostrar todas las visualizaciones", value=False)

    if show_all:
        # Mostrar todas las imágenes
        st.info(f"Mostrando {len(results)} visualizaciones...")

        for file_name, result in results.items():
            with st.expander(f"🌳 {file_name}", expanded=False):
                # Crear visualización
                fig = create_visualization(
                    result['image'],
                    result['instances'],
                    predictor.metadata,
                    file_name,
                    result['coverage']
                )
                st.pyplot(fig)
                plt.close(fig)

                # Métricas individuales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Instancias", len(result['annotations']))
                with col2:
                    individual = sum(1 for a in result['annotations'] if a['class'] == 'individual_tree')
                    st.metric("Individuales", individual)
                with col3:
                    groups = sum(1 for a in result['annotations'] if a['class'] == 'group_of_trees')
                    st.metric("Grupos", groups)
    else:
        # Mostrar selector para ver imágenes individuales
        st.write("Selecciona una imagen para ver su visualización:")
        selected_image = st.selectbox(
            "Imagen:",
            options=list(results.keys()),
            format_func=lambda x: f"{x} ({len(results[x]['annotations'])} instancias, {results[x]['coverage']:.1f}% cobertura)"
        )

        if selected_image:
            result = results[selected_image]

            # Métricas de la imagen seleccionada
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Instancias Detectadas", len(result['annotations']))
            with col2:
                individual = sum(1 for a in result['annotations'] if a['class'] == 'individual_tree')
                st.metric("Árboles Individuales", individual)
            with col3:
                groups = sum(1 for a in result['annotations'] if a['class'] == 'group_of_trees')
                st.metric("Grupos de Árboles", groups)

            # Visualización
            fig = create_visualization(
                result['image'],
                result['instances'],
                predictor.metadata,
                selected_image,
                result['coverage']
            )
            st.pyplot(fig)
            plt.close(fig)

            # Detalles de detecciones
            with st.expander("Ver detalles de detecciones"):
                for i, ann in enumerate(result['annotations'], 1):
                    st.write(f"**Detección {i}:**")
                    st.write(f"  - Clase: `{ann['class']}`")
                    st.write(f"  - Confianza: `{ann['confidence_score']}`")
                    st.write("---")

    # Descargar resultados
    st.subheader("📥 Descargar Resultados")

    # Crear archivo JSON de submisión
    submission = {
        'images': []
    }

    for file_name, result in results.items():
        submission['images'].append({
            'file_name': file_name,
            'width': result['image'].shape[1],
            'height': result['image'].shape[0],
            'annotations': result['annotations']
        })

    json_str = json.dumps(submission, indent=2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Descargar JSON",
            data=json_str,
            file_name="submission_batch.json",
            mime="application/json"
        )

    with col2:
        # CSV de estadísticas
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="statistics_batch.csv",
            mime="text/csv"
        )

    with col3:
        # Generar ZIP con todas las visualizaciones
        if st.button("🎨 Generar Visualizaciones ZIP"):
            with st.spinner("Generando visualizaciones..."):
                OUTPUT_DIR.mkdir(exist_ok=True)

                # Crear archivo ZIP en memoria
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_name, result in results.items():
                        # Crear visualización
                        fig = create_visualization(
                            result['image'],
                            result['instances'],
                            predictor.metadata,
                            file_name,
                            result['coverage']
                        )

                        # Guardar en buffer
                        img_buffer = BytesIO()
                        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        # Agregar al ZIP
                        img_name = file_name.replace('.tif', '.png').replace('.jpg', '.png').replace('.jpeg', '.png')
                        zip_file.writestr(f"visualizations/{img_name}", img_buffer.getvalue())

                zip_buffer.seek(0)

                st.download_button(
                    label="📦 Descargar ZIP de Visualizaciones",
                    data=zip_buffer,
                    file_name="visualizations_batch.zip",
                    mime="application/zip"
                )


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Interfaz principal de Streamlit."""

    # Título
    st.title("🌳 Segmentación Panóptica de Árboles")
    st.markdown("**Sistema de Deep Learning para Detección de Dosel Arbóreo Urbano**")
    st.markdown("---")

    # Verificar disponibilidad
    if not DETECTRON2_AVAILABLE:
        st.error("❌ Detectron2 no está disponible. Instala las dependencias necesarias.")
        st.stop()

    # Sidebar - Información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **Modelo:** Mask R-CNN (Detectron2)

        **Clases:**
        - 🌳 Árboles individuales
        - 🌲 Grupos de árboles

        **Métricas:**
        - Cobertura de dosel (%)
        - Número de instancias
        - Confianza de predicción
        """)

        st.markdown("---")
        st.header("⚙️ Configuración")

        # Umbral de confianza
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Filtrar detecciones con confianza menor a este umbral"
        )

        st.markdown("---")
        st.info("💡 **Formatos soportados:** .tif, .tiff, .png, .jpg, .jpeg")

    # Cargar modelo
    predictor = load_model()

    if predictor is None:
        st.stop()

    # Actualizar umbral de confianza
    if hasattr(predictor, 'cfg'):
        predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        predictor.predictor = predictor.predictor.__class__(predictor.cfg)

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📷 Imagen Única", "📁 Procesamiento por Lotes", "📊 Información General"])

    # TAB 1: Imagen única
    with tab1:
        st.header("Procesar Imagen Individual")

        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            help="Sube una imagen para procesar"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Imagen Original")
                # Mostrar preview
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, width='stretch')
                except Exception as e:
                    st.warning(f"No se puede mostrar preview: {e}")

            with col2:
                if st.button("🚀 Procesar Imagen", type="primary"):
                    process_single_image(predictor, uploaded_file)

    # TAB 2: Procesamiento por lotes
    with tab2:
        st.header("Procesamiento por Lotes")

        uploaded_files = st.file_uploader(
            "Selecciona múltiples imágenes",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Sube múltiples imágenes para procesarlas en lote"
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} archivos seleccionados")

            # Mostrar lista de archivos
            with st.expander("Ver archivos seleccionados"):
                for f in uploaded_files:
                    st.write(f"- {f.name}")

            if st.button("🚀 Procesar Lote", type="primary"):
                process_batch_images(predictor, uploaded_files)

        # Mostrar resultados si existen en session_state
        display_batch_results(predictor)

    # TAB 3: Información del modelo
    with tab3:
        st.header("Acerca del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏗️ Arquitectura")
            st.markdown("""
            **Mask R-CNN con Detectron2**

            - **Backbone:** ResNet-50 + FPN
            - **Framework:** Detectron2 (Facebook AI Research)
            - **Tarea:** Segmentación de instancias
            - **Pre-entrenamiento:** COCO Dataset
            - **Fine-tuning:** Dataset de árboles urbanos
            """)

        with col2:
            st.subheader("📊 Parámetros de Entrenamiento")
            st.markdown("""
            - **Iteraciones:** 50000
            - **Learning Rate:** 0.0001
            - **Batch Size:** 2
            - **Clases:** 2 (individual, grupo)
            - **Umbral de score:** 0.5 (configurable)
            """)

        st.markdown("---")

        st.subheader("🎯 Uso del Modelo")
        st.markdown("""
        ### Clasificación por Área

        El modelo clasifica automáticamente los árboles según su área:

        - **Árbol Individual:** 100 - 8000 píxeles
        - **Grupo de Árboles:** > 8000 píxeles

        ### Métricas de Salida

        - **Cobertura de dosel:** Porcentaje del área cubierta por vegetación
        - **Número de instancias:** Cantidad de árboles/grupos detectados
        - **Confianza:** Score de confianza de cada predicción (0-1)
        - **Segmentación:** Polígonos de las máscaras de cada instancia
        """)

        st.markdown("---")

        st.subheader("👷Grupo de Trabajo")
        st.markdown("""
        | Nombre                         |
        |--------------------------------|
        | Adriana María Ríos             |
        | Andrés Mauricio Martínez Celis |
        | Diego Alberto Rodríguez Cruz   |
        | Johana Ríos Solano             |
        """)
        
        # import torch

        # info_data = {
        #     "Ruta del modelo": str(MODEL_PATH),
        #     "Modelo existe": "✓" if MODEL_PATH.exists() else "❌",
        #     "PyTorch version": torch.__version__,
        #     "CUDA disponible": "✓" if torch.cuda.is_available() else "❌",
        # }

        # if torch.cuda.is_available():
        #     info_data["GPU"] = torch.cuda.get_device_name(0)

        # for key, value in info_data.items():
        #     st.write(f"**{key}:** {value}")


# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    main()
