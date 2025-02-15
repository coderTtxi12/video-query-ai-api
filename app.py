import os
import math
import cv2
import requests
import clip
import torch
import base64
import string
import random
import datetime
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Permitir CORS para peticiones desde cualquier dominio


# CONFIGURACIÓN GLOBAL
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Diccionario para almacenar datos de videos procesados en memoria.
videos_data = {}

# FUNCIONES AUXILIARES
def generate_id(length=8):
    """Genera un ID aleatorio para cada video."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def pil_image_to_base64(image):
    """Convierte una imagen PIL a un string Base64 (útil para retornar en JSON)."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ENDPOINT 1: PROCESAR VIDE
@app.route("/process-video", methods=["POST"])
def process_video():

    """
    1) Recibe un JSON con 'video_url' y 'N'.
    2) Descarga el video (MP4 directo) usando requests.
    3) Extrae cada N-th frame.
    4) Encodea los frames con CLIP de OPEN AI y los almacena en memoria.
    5) Devuelve un ID de video para búsquedas posteriores.
    """
    try:
        data = request.get_json()
        video_url = data["video_url"]
        N = int(data["N"])

        # 1) Descargar el video con requests
        video_id = generate_id()
        filename = f"video_{video_id}.mp4"

        # Realizamos la petición GET
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to download video. Status code: {response.status_code}"}), 400

        # Guardamos el contenido en un archivo local
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # 2) Extraer frames cada N usando OpenCV
        capture = cv2.VideoCapture(filename)
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0  # Previene división por cero si fps = 0

        video_frames = []
        current_frame = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            # Convertir a PIL (RGB)
            pil_frame = Image.fromarray(frame[:, :, ::-1])
            video_frames.append(pil_frame)

            # Avanzar N frames
            current_frame += N
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        capture.release()
        os.remove(filename)  # Borramos el video local para no ocupar espacio

        # 3) Encodear frames con CLIP
        batch_size = 64 
        batches = math.ceil(len(video_frames) / batch_size)

        # Tensor para almacenar los features de todos los frames
        all_features = torch.empty([0, 512], dtype=torch.float16).to(device)

        for i in range(batches):
            batch_frames = video_frames[i * batch_size : (i + 1) * batch_size]
            batch_preprocessed = torch.stack([preprocess(f) for f in batch_frames]).to(device)

            with torch.no_grad():
                batch_features = model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)

            all_features = torch.cat((all_features, batch_features))

        # 4) Guardar datos en memoria
        videos_data[video_id] = {
            "frames": video_frames,
            "features": all_features,
            "fps": fps,
            "N": N,
            "video_url": video_url
        }

        return jsonify({
            "video_id": video_id,
            "frames_extracted": len(video_frames)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ENDPOINT 2: BUSCAR EN EL VIDEO
@app.route("/search", methods=["POST"])
def search_in_video():
    """
    1) Recibe un JSON con 'video_id' y 'search_query'.
    2) Codifica la búsqueda (texto) con CLIP.
    3) Calcula la similitud con los frames guardados.
    4) Retorna los top K resultados, incluyendo imagen en Base64 y timestamp.
    """
    try:
        data = request.get_json()
        video_id = data["video_id"]
        search_query = data["search_query"]
        top_k = int(data.get("top_k", 3))  # Por defecto retorna los 3 primeros

        # Validar que exista el video procesado
        if video_id not in videos_data:
            return jsonify({"error": "Video ID not found. Please process the video first."}), 404

        video_info = videos_data[video_id]
        frames = video_info["frames"]
        features = video_info["features"]
        fps = video_info["fps"]
        N = video_info["N"]

        # 1) Codificar búsqueda con CLIP
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize(search_query).to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # 2) Similaridad coseno (frame_features @ text_features)
        similarities = 100.0 * features @ text_features.T  # [num_frames, 1]
        values, best_idx = similarities.topk(top_k, dim=0)

        # 3) Construir resultados
        results = []
        # best_idx es un tensor; convertimos a numpy si es necesario
        best_idx_list = best_idx.squeeze().cpu().numpy()

        # Asegurar que best_idx_list sea iterable (si top_k=1, puede ser un int)
        if isinstance(best_idx_list, np.integer):
            best_idx_list = [best_idx_list]

        for frame_id in best_idx_list:
            frame_pil = frames[frame_id]
            frame_base64 = pil_image_to_base64(frame_pil)

            # Calcular timestamp aproximado
            seconds = round(frame_id * N / fps)
            found_time = str(datetime.timedelta(seconds=seconds))

            # Extraer la similitud correspondiente
            sim_value = float(values[best_idx == frame_id].item())

            results.append({
                "frame_id": int(frame_id),
                "similarity": sim_value,
                "timestamp": found_time,
                "timestamp_seconds": seconds,
                # "image_base64": frame_base64
            })

        return jsonify({
            "video_id": video_id,
            "search_query": search_query,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# MAIN
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
