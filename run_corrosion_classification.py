# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:20:57 2026

@author: vinicius.ferreira
"""
from ultralytics import YOLO
from datetime import datetime
import os
import sys

# Extensões que o YOLO reconhece nativamente como imagem ou vídeo.
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
VALID_EXTENSIONS = IMG_EXTENSIONS | VIDEO_EXTENSIONS


def corrosion_classification(midia_path, output_folder):

    is_file = os.path.isfile(midia_path)
    is_dir = os.path.isdir(midia_path)

    if not is_file and not is_dir:
        print(f"Error: Path '{midia_path}' not found (nem arquivo nem pasta).")
        return

    if output_folder[1] != ":":
        output_folder = f"{os.getcwd()}\\{output_folder}"
        
    # Datetime na saida do filename
    today = datetime.today()
    formatted_date = today.strftime("%Y%m%d_%H%M%S")

    # Yolo trained model
    model = YOLO(r"C:/Users/vinicius.ferreira/Documents/Projects/corrosao_visualization/treino/modelos/weights/best.pt")

    if is_file:
        file_name = os.path.splitext(os.path.basename(midia_path))[0]
        file_format = os.path.splitext(os.path.basename(midia_path))[1]

        if file_format.lower() not in VALID_EXTENSIONS:
            print(f"Error: extensão '{file_format}' não suportada.")
            return

        run_name = f"{formatted_date}_{file_name}"
        print(f"Starting processing midia {file_name}{file_format}")

        results = model.predict(
            source=midia_path,
            save=True,
            conf=0.25,
            project=output_folder,
            name=run_name,
        )
        results[0].show()

    else:
        # Pasta: o YOLO processa imagens e vídeos dentro dela automaticamente
        found = [
            f for f in os.listdir(midia_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        if not found:
            print(f"Error: nenhuma imagem/vídeo suportado encontrado em '{midia_path}'.")
            return

        folder_name = os.path.basename(os.path.normpath(midia_path))
        run_name = f"{formatted_date}_{folder_name}"
        print(f"Starting processing {len(found)} arquivo(s) em '{midia_path}'")

        # stream=True evita acumular todos os resultados em RAM de uma vez (importante com pastas grandes ou vídeos longos misturados)
        results_generator = model.predict(
            source=midia_path,
            save=True,
            conf=0.25,
            project=output_folder,
            name=run_name,
            stream=True,
        )

        # Com stream=True, é preciso iterar para os resultados serem de fato processados e salvos (senão o generator nunca roda).
        for i, result in enumerate(results_generator, start=1):
            src_name = os.path.basename(result.path)
            print(f"  [{i}/{len(found)}] processado: {src_name}")

    print(f"Concluído. Resultados salvos em: {os.path.join(output_folder, run_name)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_corrosion_classification.py <midia_path_ou_pasta> <output_folder> ")
        sys.exit(1)
    midia_path = sys.argv[1]
    output_folder = sys.argv[2]
    corrosion_classification(midia_path, output_folder)


