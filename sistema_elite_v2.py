# -*- coding: utf-8 -*-
"""
====================================================================================================
SISTEMA ELITE DE TRACKING TÁCTICO FÚTBOL - VERSIÓN FINAL CERTIFICADA
====================================================================================================
 TODAS LAS CORRECCIONES FINALES APLICADAS:
 detect_formation añadido a config
 IoU real implementado (40% dist + 30% reid + 30% iou)
 cost_threshold usado correctamente
 CSV con celdas vacías en lugar de "None"
 Flush final del ensemble si no alcanza cupo completo
 self.events eliminado (no usado)
 Banner correcto v2.0.1

VERSIÓN: 2.0.1-CERTIFIED
FECHA: 2025
====================================================================================================
"""

import os
import sys
import json
import csv
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, deque
from enum import Enum

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # <- debe ir antes de importar pyplot
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle, Polygon
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, DBSCAN

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import resnet50
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ultralytics import YOLO

# --- Función auxiliar para serializar JSON robusto ---
def _json_default(o):
    import numpy as np
    from enum import Enum
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ====================================================================================================
# CONFIGURACIÓN
# ====================================================================================================
class SystemMode(Enum):
    ELITE = "elite"
    PERFORMANCE = "performance"
    BASIC = "basic"

@dataclass
class EliteConfig:
    # Video
    video_path: str = 'videos/partido.mp4'
    inicio_segundos: int = 670
    duracion_segundos: int = 10
    
    # Sistema
    mode: SystemMode = SystemMode.ELITE
    device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    # YOLO (SIN tracker externo - usamos nuestro ReID)
    yolo_model: str = 'yolov8x.pt'
    conf_threshold: float = 0.20
    iou_threshold: float = 0.45
    
    # Clasificación
    use_hsv_classifier: bool = True
    use_histogram_classifier: bool = True
    use_deep_classifier: bool = TORCH_AVAILABLE
    voting_threshold: float = 0.6
    
    # HSV (se auto-ajustan si auto_tune=True)
    color_equipo: str = 'CELESTE'
    hsv_ranges: Dict = field(default_factory=lambda: {
        'h': [60, 140], 's': [0, 255], 'v': [80, 255]
    })
    porcentaje_color_min: float = 0.10
    auto_tune_hsv: bool = True
    bootstrap_samples: int = 50  # Muestras para entrenar ensemble
    
    # Filtros
    area_min: int = 800
    area_max: int = 50000
    use_field_mask: bool = True
    margin_pixels: int = 50
    
    # ReID y Tracking
    use_reid: bool = TORCH_AVAILABLE
    reid_threshold: float = 0.65
    reid_gallery_size: int = 10
    max_disappeared: int = 20
    cost_threshold: float = 0.6  # Umbral para asignación (compara con score)
    
    # Homografía
    use_homography: bool = True
    auto_calibrate: bool = False  # Desactivado por defecto (problemas detectados)
    manual_points: Optional[List] = field(default_factory=lambda: [
        [200, 650], [1720, 650], [1480, 180], [380, 180]
    ])
    
    # Vista Táctica
    tactical_view_width: int = 800
    tactical_view_height: int = 1200
    show_voronoi: bool = True
    show_heatzone: bool = True
    show_formation: bool = True
    detect_formation: bool = True
    update_tactical_every_n_frames: int = 5
    use_kmeans_formation: bool = True  # KMeans en lugar de gaps
    
    # Outliers y self-healing
    use_dbscan_outliers: bool = True
    dbscan_eps: float = 8.0  # metros
    dbscan_min_samples: int = 2
    enable_self_healing: bool = True
    
    # Output
    save_video: bool = True
    save_tactical_video: bool = True
    save_csv_tracks: bool = True
    save_csv_events: bool = True
    save_json: bool = True
    
    # Debug
    debug_mode: bool = False
    verbose: int = 1

CONFIG = EliteConfig()

print("=" * 100)
print(" SISTEMA ELITE v2.0.1 - VERSIÓN FINAL")
print("=" * 100)
print(f"\n  Modo: {CONFIG.mode.value.upper()}")
print(f"   Device: {CONFIG.device.upper()}")
print(f"   ReID: {'✅' if CONFIG.use_reid else '❌'}")
print(f"   Homografía: {'✅' if CONFIG.use_homography else '❌'}")

# ====================================================================================================
# ESTRUCTURAS DE DATOS
# ====================================================================================================
@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    embedding: Optional[np.ndarray] = None
    team_votes: Dict[str, float] = field(default_factory=dict)
    is_my_team: bool = False
    
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)

@dataclass
class TrackedPlayer:
    track_id: int
    detections: deque = field(default_factory=lambda: deque(maxlen=30))
    embedding_gallery: deque = field(default_factory=lambda: deque(maxlen=10))
    frames_missing: int = 0
    confidence_score: float = 1.0
    
    def get_position(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        for det_frame, det in self.detections:
            if det_frame == frame_idx:
                return det.center()
        return None
    
    def get_last_position(self) -> Optional[Tuple[float, float]]:
        if len(self.detections) > 0:
            return self.detections[-1][1].center()
        return None

class TacticalSnapshot:
    def __init__(self, frame_idx: int, players: List[TrackedPlayer]):
        self.frame_idx = frame_idx
        self.players = players
        self.positions = []  # En METROS (no píxeles)
        
        self.formation: Optional[str] = None
        self.compactness: float = 0.0
        self.width: float = 0.0
        self.height: float = 0.0
        self.centroid: Optional[Tuple[float, float]] = None
    
    def calculate_metrics(self, use_kmeans: bool = True):
        if len(self.positions) < 3:
            return
        
        pos_array = np.array(self.positions)
        
        # Centroide
        self.centroid = tuple(pos_array.mean(axis=0))
        
        # Compactness (metros)
        distances = np.linalg.norm(pos_array - self.centroid, axis=1)
        self.compactness = float(distances.mean())
        
        # Ancho y alto (metros)
        self.width = float(pos_array[:, 0].max() - pos_array[:, 0].min())
        self.height = float(pos_array[:, 1].max() - pos_array[:, 1].min())
        
        # Formación
        self.formation = self._detect_formation_kmeans(pos_array) if use_kmeans else self._detect_formation_gaps(pos_array)
    
    def _detect_formation_kmeans(self, positions: np.ndarray) -> str:
        """Detección robusta con KMeans"""
        if len(positions) < 7:
            return "INCOMPLETE"
        
        try:
            # Clustering en eje Y (profundidad)
            kmeans = KMeans(n_clusters=min(4, len(positions)), random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions[:, 1].reshape(-1, 1))
            
            # Ordenar clusters por profundidad
            cluster_y = []
            for k in range(kmeans.n_clusters):
                cluster_points = positions[labels == k]
                if len(cluster_points) > 0:
                    cluster_y.append((k, cluster_points[:, 1].mean(), len(cluster_points)))
            
            cluster_y.sort(key=lambda x: x[1])  # Ordenar por Y
            
            # Tomar últimas 3 líneas (defensa, medio, ataque)
            line_counts = [count for _, _, count in cluster_y[-3:]]
            
            return "-".join(map(str, line_counts))
        except:
            return "UNKNOWN"
    
    def _detect_formation_gaps(self, positions: np.ndarray) -> str:
        """Método legacy por gaps"""
        sorted_y = np.sort(positions[:, 1])
        gaps = np.diff(sorted_y)
        threshold = gaps.mean() + gaps.std()
        lines = np.where(gaps > threshold)[0] + 1
        
        line_counts = []
        prev = 0
        for line_idx in np.append(lines, len(sorted_y)):
            line_counts.append(line_idx - prev)
            prev = line_idx
        
        if len(line_counts) >= 3:
            return "-".join(map(str, line_counts[-3:]))
        return "UNKNOWN"

# ====================================================================================================
# EXTRACTOR DE FEATURES COMPARTIDO (Optimización memoria)
# ====================================================================================================
class SharedFeatureExtractor:
    """ResNet50 compartido para DeepClassifier y ReID"""
    _instance = None
    
    def __new__(cls, device='cpu'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, device='cpu'):
        if not self.initialized and TORCH_AVAILABLE:
            self.device = device
            self.model = resnet50(weights='IMAGENET1K_V1')
            self.model.fc = nn.Identity()
            self.model.eval()
            self.model.to(device)
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.initialized = True
            print(" Feature extractor compartido inicializado")
    
    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop.size == 0:
            return None
        
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.cpu().numpy().flatten()

# ====================================================================================================
# CALIBRACIÓN CON FALLBACK ROBUSTO
# ====================================================================================================
class AutoFieldCalibrator:
    def __init__(self, config: EliteConfig):
        self.config = config
        self.H = None
        self.H_inv = None
        self.field_mask = None
        self.calibrated = False
        
        self.field_real = np.array([
            [0, 0], [105, 0], [105, 68], [0, 68]
        ], dtype=np.float32)
    
    def calibrate_manual(self, points: List[List[int]]):
        """Calibración manual robusta"""
        try:
            pts = np.array(points, dtype=np.float32)
            if pts.shape != (4, 2):
                raise ValueError("Se necesitan exactamente 4 puntos [x,y]")
            
            self.H, _ = cv2.findHomography(pts, self.field_real, method=cv2.RANSAC)
            self.H_inv, _ = cv2.findHomography(self.field_real, pts, method=cv2.RANSAC)
            
            if self.H is None or self.H_inv is None:
                raise ValueError("Homografía no calculable")
            
            self.calibrated = True
            print(" Calibración manual exitosa")
        except Exception as e:
            print(f" Error en calibración manual: {e}")
            self.calibrated = False
    
    def create_field_mask(self, frame_shape: Tuple) -> np.ndarray:
        """Máscara del campo con margen"""
        if not self.calibrated:
            return np.ones((frame_shape[0], frame_shape[1]), dtype=np.uint8) * 255
        
        # Campo expandido (margen de 5m)
        field_expanded = self.field_real + np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
        field_pixels = cv2.perspectiveTransform(
            field_expanded.reshape(1, -1, 2), self.H_inv
        )[0]
        
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [field_pixels.astype(np.int32)], 255)
        
        self.field_mask = mask
        return mask
    
    def pixel_to_real(self, x: float, y: float) -> Tuple[float, float]:
        """Píxeles → Metros"""
        if not self.calibrated:
            return (x, y)
        
        pt = np.array([[[x, y]]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, self.H)
        return tuple(real[0][0])

# ====================================================================================================
# CLASIFICADORES CON TRAINING ONLINE
# ====================================================================================================
class HSVClassifier:
    name = "HSV"
    
    def __init__(self, config: EliteConfig):
        self.config = config
        self.ranges = config.hsv_ranges
        self.samples = []  # Para auto-tune
    
    def predict(self, crop: np.ndarray) -> Tuple[bool, float]:
        h = crop.shape[0]
        torso = crop[int(h*0.20):int(h*0.60), :]
        
        if torso.size == 0:
            return False, 0.0
        
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        lower = np.array([self.ranges['h'][0], self.ranges['s'][0], self.ranges['v'][0]], dtype=np.uint8)
        upper = np.array([self.ranges['h'][1], self.ranges['s'][1], self.ranges['v'][1]], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Limpieza morfológica para luces duras
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        percentage = cv2.countNonZero(mask) / max(mask.size, 1)
        
        is_team = percentage >= self.config.porcentaje_color_min
        confidence = min(percentage / 0.3, 1.0)
        
        # Guardar muestra para auto-tune
        if is_team and confidence > 0.7 and len(self.samples) < 100:
            self.samples.append(hsv)
        
        return is_team, confidence
    
    def auto_tune(self):
        """Ajusta rangos HSV con percentiles por canal"""
        if len(self.samples) < 20:
            return
        
        # Extraer canales y aplanar
        H_vals, S_vals, V_vals = [], [], []
        for hsv in self.samples:
            H_vals.append(hsv[..., 0].ravel())
            S_vals.append(hsv[..., 1].ravel())
            V_vals.append(hsv[..., 2].ravel())
        
        H_vals = np.concatenate(H_vals)
        S_vals = np.concatenate(S_vals)
        V_vals = np.concatenate(V_vals)
        
        h_min, h_max = np.percentile(H_vals, [10, 90]).astype(int)
        s_min, s_max = np.percentile(S_vals, [10, 90]).astype(int)
        v_min, v_max = np.percentile(V_vals, [10, 90]).astype(int)
        
        # Asegurar rangos válidos y protección para equipaciones fluorescentes
        h_min, h_max = max(0, h_min), min(179, h_max)
        s_min, s_max = max(0, s_min), min(255, s_max)
        v_min, v_max = max(60, v_min), min(255, v_max)  # v_min >= 60 para colores vivos
        
        self.ranges = {'h': [h_min, h_max], 's': [s_min, s_max], 'v': [v_min, v_max]}
        print(f"✅ HSV auto-tuned: H{self.ranges['h']} S{self.ranges['s']} V{self.ranges['v']}")

class HistogramClassifier:
    name = "Histogram"
    
    def __init__(self, config: EliteConfig):
        self.config = config
        self.reference_hist = None
        self.trained = False
    
    def train(self, crops: List[np.ndarray]):
        hists = []
        for crop in crops:
            if crop.size == 0:
                continue
            h = crop.shape[0]
            torso = crop[int(h*0.2):int(h*0.6), :]
            if torso.size == 0:
                continue
            
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hists.append(hist)
        
        if hists:
            self.reference_hist = np.mean(hists, axis=0)
            self.trained = True
    
    def predict(self, crop: np.ndarray) -> Tuple[bool, float]:
        if not self.trained:
            return False, 0.0
        
        h = crop.shape[0]
        torso = crop[int(h*0.2):int(h*0.6), :]
        if torso.size == 0:
            return False, 0.0
        
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        similarity = cv2.compareHist(self.reference_hist.astype(np.float32), 
                                    hist.astype(np.float32), cv2.HISTCMP_CORREL)
        
        return similarity > 0.6, max(0, similarity)

class DeepClassifier:
    name = "Deep"
    
    def __init__(self, config: EliteConfig, feature_extractor: SharedFeatureExtractor):
        self.config = config
        self.extractor = feature_extractor
        self.reference_embedding = None
        self.trained = False
    
    def train(self, crops: List[np.ndarray]):
        embeddings = []
        for crop in crops:
            if crop.size == 0:
                continue
            h = crop.shape[0]
            torso = crop[int(h*0.2):int(h*0.6), :]
            if torso.size == 0:
                continue
            
            emb = self.extractor.extract(torso)
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            self.reference_embedding = np.mean(embeddings, axis=0)
            self.trained = True
    
    def predict(self, crop: np.ndarray) -> Tuple[bool, float]:
        if not self.trained:
            return False, 0.0
        
        h = crop.shape[0]
        torso = crop[int(h*0.2):int(h*0.6), :]
        if torso.size == 0:
            return False, 0.0
        
        emb = self.extractor.extract(torso)
        if emb is None:
            return False, 0.0
        
        similarity = np.dot(self.reference_embedding, emb) / (
            np.linalg.norm(self.reference_embedding) * np.linalg.norm(emb) + 1e-8
        )
        
        return similarity > 0.7, max(0, similarity)

class EnsembleTeamClassifier:
    def __init__(self, config: EliteConfig, feature_extractor: SharedFeatureExtractor):
        self.config = config
        self.classifiers = []
        
        if config.use_hsv_classifier:
            self.classifiers.append(HSVClassifier(config))
        if config.use_histogram_classifier:
            self.classifiers.append(HistogramClassifier(config))
        if config.use_deep_classifier and TORCH_AVAILABLE:
            self.classifiers.append(DeepClassifier(config, feature_extractor))
        
        print(f" Ensemble: {len(self.classifiers)} clasificadores")
    
    def classify(self, frame: np.ndarray, detection: Detection) -> bool:
        votes = []
        x1, y1, x2, y2 = detection.bbox.astype(int)
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        
        if crop.size == 0:
            return False
        
        for clf in self.classifiers:
            vote, conf = clf.predict(crop)
            votes.append((vote, conf))
            detection.team_votes[clf.name] = conf if vote else -conf
        
        weighted_sum = sum(conf if vote else -conf for vote, conf in votes)
        total_conf = sum(abs(conf) for _, conf in votes)
        
        # Fallback a HSV puro si ensemble no entrenado (total_conf muy bajo)
        if total_conf < 0.1:
            for clf in self.classifiers:
                if clf.name == "HSV":
                    vote, _ = clf.predict(crop)
                    return vote
            return False
        
        normalized = weighted_sum / total_conf
        return normalized > (2 * self.config.voting_threshold - 1)

# ====================================================================================================
# REID SIMPLIFICADO CON RECOVERY
# ====================================================================================================
class EliteReID:
    def __init__(self, config: EliteConfig, feature_extractor: SharedFeatureExtractor):
        self.config = config
        self.extractor = feature_extractor
        self.gallery: Dict[int, deque] = defaultdict(lambda: deque(maxlen=config.reid_gallery_size))
        self.active = TORCH_AVAILABLE
    
    def extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        if not self.active:
            return None
        
        x1, y1, x2, y2 = bbox.astype(int)
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        
        return self.extractor.extract(crop)
    
    def update_gallery(self, track_id: int, embedding: np.ndarray):
        self.gallery[track_id].append(embedding)
    
    def get_average_embedding(self, track_id: int) -> Optional[np.ndarray]:
        if track_id not in self.gallery or len(self.gallery[track_id]) == 0:
            return None
        return np.mean(list(self.gallery[track_id]), axis=0)
    
    def match(self, embedding: np.ndarray, existing_ids: Set[int]) -> Optional[int]:
        best_sim = -1
        best_id = None
        
        for tid in existing_ids:
            avg_emb = self.get_average_embedding(tid)
            if avg_emb is None:
                continue
            
            sim = np.dot(avg_emb, embedding) / (
                np.linalg.norm(avg_emb) * np.linalg.norm(embedding) + 1e-8
            )
            
            if sim > best_sim:
                best_sim = sim
                best_id = tid
        
        return best_id if best_sim > self.config.reid_threshold else None

# ====================================================================================================
# UTILS: IoU
# ====================================================================================================
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union

# ====================================================================================================
# VISTA TÁCTICA CON LIMPIEZA CORRECTA
# ====================================================================================================
# ====================================================================================================
# VISTA TÁCTICA CON LIMPIEZA CORRECTA (COMPATIBLE CON ArtistList)
# ====================================================================================================
class TacticalViewRenderer:
    def __init__(self, config: EliteConfig):
        self.config = config
        self.w = config.tactical_view_width
        self.h = config.tactical_view_height
        self.fig = None
        self.ax = None
        self.static_patches_count = 0
        self.static_lines_count = 0
        self.static_collections_count = 0
        plt.ioff()

    def setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(self.w / 100, self.h / 100), dpi=100)
        self.ax.set_xlim(0, 105)
        self.ax.set_ylim(0, 68)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self._draw_field()

        # Guardar cuántos elementos estáticos hay tras dibujar el campo
        self.static_patches_count = len(self.ax.patches)
        self.static_lines_count = len(self.ax.lines)
        self.static_collections_count = len(self.ax.collections)

    def _draw_field(self):
        # Perímetro
        self.ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, edgecolor='white', linewidth=2))
        # Línea central
        self.ax.plot([52.5, 52.5], [0, 68], 'w-', linewidth=2)
        # Círculo central
        self.ax.add_patch(Circle((52.5, 34), 9.15, fill=False, edgecolor='white', linewidth=2))

        # Áreas
        self.ax.add_patch(Rectangle((0, 13.84), 16.5, 40.32, fill=False, edgecolor='white', linewidth=2))
        self.ax.add_patch(Rectangle((105 - 16.5, 13.84), 16.5, 40.32, fill=False, edgecolor='white', linewidth=2))
        self.ax.add_patch(Rectangle((0, 24.84), 5.5, 18.32, fill=False, edgecolor='white', linewidth=2))
        self.ax.add_patch(Rectangle((105 - 5.5, 24.84), 5.5, 18.32, fill=False, edgecolor='white', linewidth=2))

        # Puntos de penalty (dos llamadas separadas)
        self.ax.plot(11, 34, 'wo', markersize=5)
        self.ax.plot(94, 34, 'wo', markersize=5)

        self.ax.set_facecolor('#1a5f1a')
        self.fig.patch.set_facecolor('#0d0d0d')

    def render(self, snapshot: TacticalSnapshot) -> np.ndarray:
        if self.fig is None or self.ax is None:
            self.setup_canvas()

        # --- LIMPIEZA DINÁMICA ROBUSTA (no usar .pop()/.clear() sobre ArtistList) ---
        # Patches añadidos después del campo
        for art in list(self.ax.patches[self.static_patches_count:]):
            try:
                art.remove()
            except Exception:
                pass

        # Líneas añadidas después de las del campo
        for art in list(self.ax.lines[self.static_lines_count:]):
            try:
                art.remove()
            except Exception:
                pass

        # Colecciones (scatter, contourf, etc.) añadidas dinámicamente
        for art in list(self.ax.collections[self.static_collections_count:]):
            try:
                art.remove()
            except Exception:
                pass

        # Textos dinámicos
        for txt in list(self.ax.texts):
            try:
                txt.remove()
            except Exception:
                pass
        # --- FIN LIMPIEZA ---

        if not snapshot.positions:
            return self._fig_to_array()

        positions = np.array(snapshot.positions)

        # Dibujar jugadores
        self.ax.scatter(
            positions[:, 0], positions[:, 1],
            c='cyan', s=200, edgecolors='white', linewidths=2, zorder=10
        )

        # Etiquetas 1..N
        for i, (x, y) in enumerate(positions):
            self.ax.text(
                x, y, str(i + 1),
                ha='center', va='center',
                color='black', fontsize=8, weight='bold', zorder=11
            )

        # Centroide
        if snapshot.centroid:
            self.ax.plot(snapshot.centroid[0], snapshot.centroid[1], 'r*', markersize=15, zorder=9)

        # Voronoi
        if getattr(self.config, "show_voronoi", True) and len(positions) >= 4:
            try:
                vor = Voronoi(positions)
                for region_idx in vor.point_region:
                    region = vor.regions[region_idx]
                    if region and -1 not in region:
                        polygon = [vor.vertices[i] for i in region]
                        self.ax.add_patch(Polygon(polygon, alpha=0.15, facecolor='cyan', edgecolor='cyan'))
            except Exception:
                pass

        # Heatzone
        if getattr(self.config, "show_heatzone", True):
            x = np.linspace(0, 105, 50)
            y = np.linspace(0, 68, 32)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X, dtype=float)
            for px, py in positions:
                dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
                Z += np.exp(-dist / 8.0)
            Z = gaussian_filter(Z, sigma=1.5)
            self.ax.contourf(X, Y, Z, levels=10, cmap='Reds', alpha=0.3, zorder=1)

        # Métricas
        if getattr(self.config, "show_formation", True) and snapshot.formation:
            self.ax.text(52.5, 2, f"Formación: {snapshot.formation}",
                         ha='center', color='yellow', fontsize=10, weight='bold')

        self.ax.text(2, 66, f"Compactness: {snapshot.compactness:.1f}m", color='white', fontsize=9)
        self.ax.text(2, 63, f"Ancho: {snapshot.width:.1f}m", color='white', fontsize=9)
        self.ax.text(2, 60, f"Alto: {snapshot.height:.1f}m", color='white', fontsize=9)

        return self._fig_to_array()

    def _fig_to_array(self) -> np.ndarray:
        # Convertir figura de Matplotlib a imagen BGR (Agg/TkAgg)
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()

        if hasattr(self.fig.canvas, "tostring_rgb"):
            # Backends tipo Agg
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_rgb = buf.reshape(h, w, 3)
        else:
            # TkAgg u otros: usar ARGB y descartar alfa
            buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = buf.reshape(h, w, 4)
            img_rgb = img_argb[:, :, 1:4]  # ARGB -> RGB

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def cleanup(self):
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None
            self.ax = None
# ====================================================================================================
class EliteTrackingSystem:
    def __init__(self, config: EliteConfig):
        self.config = config

        # Feature extractor compartido
        self.feature_extractor = SharedFeatureExtractor(config.device) if TORCH_AVAILABLE else None

        # Componentes
        self.yolo = YOLO(config.yolo_model)
        self.calibrator = AutoFieldCalibrator(config)
        self.classifier = EnsembleTeamClassifier(config, self.feature_extractor)
        self.reid = EliteReID(config, self.feature_extractor) if config.use_reid else None
        self.tactical_view = TacticalViewRenderer(config)

        # Estado
        self.tracked_players: Dict[int, TrackedPlayer] = {}
        self.next_id = 1
        self.frame_count = 0

        # Bootstrap de training
        self._bootstrap_crops: List[np.ndarray] = []
        self._ensemble_trained = False

        # Historial
        self.tactical_snapshots: List[TacticalSnapshot] = []

        print("\n Sistema Elite v2.0.1 inicializado")

    # ----------------------------------------------------------------------------------------
    # Ciclo principal
    # ----------------------------------------------------------------------------------------
    def process_video(self):
        print("\n" + "=" * 100)
        print("🚀 PROCESAMIENTO")
        print("=" * 100)

        # 1) Calibración
        if self.config.use_homography:
            self._calibrate()

        # 2) Abrir vídeo
        cap = cv2.VideoCapture(self.config.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_inicio = int(self.config.inicio_segundos * fps)
        frames_total = min(int(self.config.duracion_segundos * fps),
                           int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - frame_inicio)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

        # 3) Outputs
        os.makedirs('results/elite', exist_ok=True)

        out_video = None
        out_tactical = None

        if self.config.save_video:
            out_video = cv2.VideoWriter(
                'results/elite/tracking_v2.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height)
            )

        if self.config.save_tactical_video:
            out_tactical = cv2.VideoWriter(
                'results/elite/tactical_v2.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (self.config.tactical_view_width, self.config.tactical_view_height)
            )

        # 4) Máscara de campo
        field_mask = self.calibrator.create_field_mask((height, width))

        print(f"\n🎬 Procesando {frames_total} frames...")

        # 5) Loop principal
        for frame_idx in tqdm(range(frames_total), desc="Tracking"):
            ret, frame = cap.read()
            if not ret:
                break

            # Detección (sin tracker externo)
            results = self.yolo.predict(
                frame, classes=[0],
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False
            )

            detections = self._process_detections(frame, results[0], field_mask)

            # Tracking propio (IoU + Dist + ReID)
            self._update_tracking(frame, detections, frame_idx)

            # Render
            frame_viz = self._render_frame(frame, frame_idx)
            if out_video:
                out_video.write(frame_viz)

            # Vista táctica
            if frame_idx % self.config.update_tactical_every_n_frames == 0:
                snapshot = self._create_snapshot(frame_idx)
                tactical_img = self.tactical_view.render(snapshot)
                if out_tactical:
                    # replicamos para mantener FPS
                    for _ in range(self.config.update_tactical_every_n_frames):
                        out_tactical.write(tactical_img)

            # Auto-tune HSV (a mitad del vídeo)
            if frame_idx == frames_total // 2 and self.config.auto_tune_hsv:
                for clf in self.classifier.classifiers:
                    if hasattr(clf, 'auto_tune'):
                        clf.auto_tune()

            self.frame_count += 1

        # 6) Cierre
        cap.release()
        if out_video:
            out_video.release()
        if out_tactical:
            out_tactical.release()

        self.tactical_view.cleanup()

        # 7) Flush final del ensemble si hizo falta
        if (not self._ensemble_trained) and len(self._bootstrap_crops) >= max(10, self.config.bootstrap_samples // 3):
            for clf in self.classifier.classifiers:
                if hasattr(clf, 'train'):
                    clf.train(self._bootstrap_crops)
            self._ensemble_trained = True
            print(f" Ensemble entrenado con {len(self._bootstrap_crops)} muestras (flush final)")

        # 8) Export
        self._export_results(fps)

        print("\n Completado")

    # ----------------------------------------------------------------------------------------
    # Calibración
    # ----------------------------------------------------------------------------------------
    def _calibrate(self):
        print("\n Calibrando...")
        if self.config.auto_calibrate:
            print("  Auto-calibración experimental - usando manual por defecto")

        if self.config.manual_points and len(self.config.manual_points) == 4:
            self.calibrator.calibrate_manual(self.config.manual_points)
        else:
            print("  Sin puntos de calibración - continuando sin homografía")
            self.config.use_homography = False

    # ----------------------------------------------------------------------------------------
    # Detecciones + clasificación
    # ----------------------------------------------------------------------------------------
    def _process_detections(self, frame: np.ndarray, results, field_mask: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Área
            area = (x2 - x1) * (y2 - y1)
            if area < self.config.area_min or area > self.config.area_max:
                continue

            # Máscara de campo
            if self.config.use_field_mask:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if (cx < 0 or cx >= field_mask.shape[1] or
                    cy < 0 or cy >= field_mask.shape[0] or
                        field_mask[cy, cx] == 0):
                    continue

            det = Detection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=conf
            )

            # Clasificación de equipo
            det.is_my_team = self.classifier.classify(frame, det)

            if det.is_my_team:
                # Embedding para ReID
                if self.reid and self.reid.active:
                    det.embedding = self.reid.extract_embedding(frame, det.bbox)

                # Bootstrap para entrenar ensemble
                self._bootstrap_team_models(frame, det)

                detections.append(det)

        return detections

    def _bootstrap_team_models(self, frame: np.ndarray, det: Detection):
        if self._ensemble_trained:
            return

        hsv_conf = det.team_votes.get("HSV", 0)
        if hsv_conf >= 0.75:
            x1, y1, x2, y2 = det.bbox.astype(int)
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0 and len(self._bootstrap_crops) < self.config.bootstrap_samples:
                self._bootstrap_crops.append(crop)

            if len(self._bootstrap_crops) == self.config.bootstrap_samples:
                for clf in self.classifier.classifiers:
                    if hasattr(clf, 'train'):
                        clf.train(self._bootstrap_crops)
                self._ensemble_trained = True
                print(" Ensemble entrenado online")

    # ----------------------------------------------------------------------------------------
    # Tracking (IoU + Distancia + ReID) con Hungarian
    # ----------------------------------------------------------------------------------------
    @staticmethod
    def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
        a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
        union = a1 + a2 - inter + 1e-6
        return float(inter / union)

    def _update_tracking(self, frame: np.ndarray, detections: List[Detection], frame_idx: int):
        assigned: Set[int] = set()
        unmatched_detections: List[Detection] = []

        active_ids = [tid for tid, p in self.tracked_players.items()
                      if p.frames_missing < self.config.max_disappeared]

        if len(active_ids) > 0 and len(detections) > 0:
            # cost = 1 - score; score = 0.4*dist_norm + 0.3*reid + 0.3*iou
            cost_matrix = np.ones((len(detections), len(active_ids)), dtype=np.float32)

            # Prepara últimos bboxes por ID
            last_bboxes = {}
            last_centers = {}
            for tid in active_ids:
                player = self.tracked_players[tid]
                if len(player.detections) > 0:
                    last_det = player.detections[-1][1]
                    last_bboxes[tid] = last_det.bbox
                    last_centers[tid] = last_det.center()
                else:
                    last_bboxes[tid] = None
                    last_centers[tid] = None

            for i, det in enumerate(detections):
                det_center = det.center()
                for j, tid in enumerate(active_ids):
                    last_center = last_centers[tid]
                    last_bbox = last_bboxes[tid]
                    if last_center is None or last_bbox is None:
                        continue

                    # Distancia normalizada (más cerca => mejor)
                    dist = np.hypot(det_center[0] - last_center[0],
                                    det_center[1] - last_center[1])
                    dist_score = 1.0 / (1.0 + dist / 100.0)

                    # IoU real
                    iou = self._bbox_iou(det.bbox, last_bbox)

                    # ReID (match binario → score 1.0 si coincide)
                    reid_score = 0.5
                    if self.reid and self.reid.active and det.embedding is not None:
                        matched_id = self.reid.match(det.embedding, {tid})
                        if matched_id == tid:
                            reid_score = 1.0

                    score = 0.4 * dist_score + 0.3 * reid_score + 0.3 * iou
                    cost_matrix[i, j] = 1.0 - score  # menor es mejor

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                score = 1.0 - float(cost_matrix[i, j])
                if score >= self.config.cost_threshold:
                    tid = active_ids[j]
                    player = self.tracked_players[tid]
                    player.detections.append((frame_idx, detections[i]))
                    player.frames_missing = 0
                    assigned.add(tid)

                    # Actualizar galería ReID
                    if self.reid and self.reid.active and detections[i].embedding is not None:
                        self.reid.update_gallery(tid, detections[i].embedding)
                else:
                    unmatched_detections.append(detections[i])

            # Cualquier detección no seleccionada por Hungarian queda sin asignar
            assigned_rows = set(row_ind.tolist())
            for i in range(len(detections)):
                if i not in assigned_rows:
                    unmatched_detections.append(detections[i])
        else:
            unmatched_detections = detections

        # Crear nuevos tracks para no asignadas
        for det in unmatched_detections:
            p = TrackedPlayer(track_id=self.next_id)
            p.detections.append((frame_idx, det))
            self.tracked_players[self.next_id] = p
            assigned.add(self.next_id)

            if self.reid and self.reid.active and det.embedding is not None:
                self.reid.update_gallery(self.next_id, det.embedding)

            self.next_id += 1

        # Incrementar missing y purgar viejos
        for tid, player in list(self.tracked_players.items()):
            if tid not in assigned:
                player.frames_missing += 1
                if player.frames_missing > self.config.max_disappeared:
                    del self.tracked_players[tid]

    # ----------------------------------------------------------------------------------------
    # Render
    # ----------------------------------------------------------------------------------------
    def _render_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        frame_viz = frame.copy()

        for tid, player in self.tracked_players.items():
            pos = player.get_position(frame_idx)
            if pos is None:
                continue

            if len(player.detections) > 0:
                _, last_det = player.detections[-1]
                x1, y1, x2, y2 = last_det.bbox.astype(int)
                color = self._get_color_for_id(tid)
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_viz, f"ID:{tid} {last_det.confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        num_active = len([p for p in self.tracked_players.values() if p.frames_missing == 0])
        cv2.putText(frame_viz, f"Jugadores: {num_active}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return frame_viz

    # ----------------------------------------------------------------------------------------
    # Snapshot táctico
    # ----------------------------------------------------------------------------------------
    def _create_snapshot(self, frame_idx: int) -> TacticalSnapshot:
        active_players = [p for p in self.tracked_players.values() if p.frames_missing == 0]
        snapshot = TacticalSnapshot(frame_idx, active_players)

        # píxeles → metros (si calibrado)
        if self.calibrator.calibrated:
            positions_m = []
            for p in active_players:
                pos_px = p.get_position(frame_idx)
                if pos_px is None:
                    continue
                x_m, y_m = self.calibrator.pixel_to_real(*pos_px)
                positions_m.append((x_m, y_m))
            snapshot.positions = positions_m
        else:
            # sin homografía, usar píxeles y evitar métricas “físicas”
            snapshot.positions = [p.get_position(frame_idx) for p in active_players
                                  if p.get_position(frame_idx) is not None]

        # Filtrar outliers (solo si hay metros)
        if (self.config.use_dbscan_outliers and
                len(snapshot.positions) >= 3 and self.calibrator.calibrated):
            X = np.array(snapshot.positions)
            db = DBSCAN(eps=self.config.dbscan_eps,
                        min_samples=self.config.dbscan_min_samples).fit(X)
            labels = db.labels_
            if np.any(labels != -1):
                unique, counts = np.unique(labels[labels != -1], return_counts=True)
                if len(unique) > 0:
                    main_cluster = unique[np.argmax(counts)]
                    snapshot.positions = [tuple(p) for p, lb in zip(X, labels) if lb == main_cluster]

        # Métricas tácticas
        if self.config.detect_formation:
            snapshot.calculate_metrics(use_kmeans=self.config.use_kmeans_formation)

        self.tactical_snapshots.append(snapshot)
        return snapshot

    @staticmethod
    def _get_color_for_id(track_id: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(seed=track_id)
        return tuple(int(x) for x in rng.integers(50, 255, size=3))

    # ----------------------------------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------------------------------
    def _export_results(self, fps: int):
        print("\n💾 Exportando...")

        os.makedirs('results/elite/data', exist_ok=True)

        # JSON
        if self.config.save_json:
            data = {
                'config': asdict(self.config),
                'players': {
                    tid: {
                        'total_frames': len(p.detections),
                        'trajectory_pixels': [
                            (int(f), float(d.center()[0]), float(d.center()[1]))
                            for f, d in p.detections
                        ]
                    }
                    for tid, p in self.tracked_players.items()
                },
                'tactical_snapshots': [
                    {
                        'frame': s.frame_idx,
                        'formation': s.formation,
                        'compactness_m': float(s.compactness),
                        'width_m': float(s.width),
                        'height_m': float(s.height),
                        'positions_m': [(float(x), float(y)) for x, y in s.positions]
                    }
                    for s in self.tactical_snapshots[::10]
                ]
            }
            with open('results/elite/data/tracking_v2.json', 'w') as f:
                json.dump(data, f, indent=2, default=_json_default)


        # CSV tracks
        if self.config.save_csv_tracks:
            with open('results/elite/data/tracks.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'id', 'x_pixel', 'y_pixel', 'x_m', 'y_m', 'confidence'])
                for tid, player in self.tracked_players.items():
                    for frame_idx, det in player.detections:
                        x_px, y_px = det.center()
                        # celdas vacías si no hay calibración
                        x_m, y_m = ("", "")
                        if self.calibrator.calibrated:
                            x_m, y_m = self.calibrator.pixel_to_real(x_px, y_px)
                        writer.writerow([frame_idx, tid, x_px, y_px, x_m, y_m, det.confidence])

        # CSV events (snapshot por frame táctico)
        if self.config.save_csv_events:
            with open('results/elite/data/events.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'timestamp_s', 'num_players', 'formation',
                                 'compactness_m', 'width_m', 'height_m'])
                for s in self.tactical_snapshots:
                    writer.writerow([
                        s.frame_idx, s.frame_idx / fps, len(s.positions),
                        s.formation, s.compactness, s.width, s.height
                    ])

        print("✅ Exportación completa")

    # ----------------------------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------------------------
    def cleanup(self):
        self.tactical_view.cleanup()

# ====================================================================================================
# MAIN
# ====================================================================================================
def main():
    system = EliteTrackingSystem(CONFIG)
    
    try:
        system.process_video()
        system.cleanup()
        
        print("\n" + "=" * 100)
        print(" SISTEMA ELITE v2.0.1-CERTIFIED COMPLETADO")
        print("=" * 100)
        print("\n Resultados:")
        print("   - results/elite/tracking_v2.mp4")
        print("   - results/elite/tactical_v2.mp4")
        print("   - results/elite/data/tracking_v2.json")
        print("   - results/elite/data/tracks.csv")
        print("   - results/elite/data/events.csv")
        print("=" * 100)
        
    except KeyboardInterrupt:
        print("\n  Interrumpido")
        system.cleanup()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        system.cleanup()

if __name__ == "__main__":
    main()

