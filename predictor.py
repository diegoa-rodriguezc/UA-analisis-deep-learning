"""
================================================================================
PREDICTOR PARA STREAMLIT - Segmentación Panóptica de Árboles
================================================================================
"""

from pathlib import Path
from typing import Dict
import numpy as np
import cv2
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Verificar si Detectron2 está instalado
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog

    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("⚠️ Detectron2 no está instalado")


class TreePanopticPredictor:
    """Predictor para segmentación panóptica de árboles."""

    def __init__(self, model_path: Path = None):
        """
        Inicializa predictor.

        Args:
            model_path: Ruta al modelo entrenado (.pth)
        """
        if not DETECTRON2_AVAILABLE:
            raise RuntimeError("Detectron2 no está instalado")

        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'model_final.pth'

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        # Configurar modelo
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.cfg = cfg
        self.predictor = DefaultPredictor(self.cfg)

        # Metadata
        self.metadata = MetadataCatalog.get("trees_predict")
        self.metadata.thing_classes = ["individual_tree", "group_of_trees"]

        self.model_info = {
            'model_path': str(model_path),
            'device': cfg.MODEL.DEVICE,
            'num_classes': 2
        }

    def predict(self, image_path: Path) -> Dict:
        """
        Predice instancias en una imagen.

        Args:
            image_path: Ruta a imagen

        Returns:
            Dict con resultados
        """
        # Cargar imagen
        im = cv2.imread(str(image_path))
        if im is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")

        # Predecir
        outputs = self.predictor(im)
        instances = outputs["instances"].to("cpu")

        # Convertir a formato JSON
        annotations = []
        for i in range(len(instances)):
            # Extraer máscara
            mask = instances.pred_masks[i].numpy().astype(np.uint8)

            # Convertir máscara a polígono
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            # Tomar contorno más grande
            contour = max(contours, key=cv2.contourArea)

            # Simplificar polígono
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            segmentation = approx_contour.reshape(-1).tolist()

            if len(segmentation) < 6:  # Al menos 3 vértices
                continue

            # Obtener clase y score
            class_id = int(instances.pred_classes[i])
            class_name = self.metadata.thing_classes[class_id]
            score = float(instances.scores[i])

            annotations.append({
                'class': class_name,
                'confidence_score': round(score, 2),
                'segmentation': segmentation
            })

        # Calcular cobertura
        coverage = 0.0
        if len(instances) > 0:
            combined_mask = instances.pred_masks.any(dim=0).numpy().astype(np.uint8)
            coverage = (combined_mask.sum() / combined_mask.size) * 100

        return {
            'file_name': image_path.name,
            'image': im,
            'annotations': annotations,
            'coverage': coverage,
            'instances': instances
        }
