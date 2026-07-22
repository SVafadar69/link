import degirum as dg 
import degirum_tools 
import lancedb
from lancedb.pydantic import LanceModel, Vector 
import uuid 
import numpy as np 
from typing import List, Dict 
import logging 
from typing import Any 
import degirum as dg 
import degirum_tools
from degirum_tools import get_video_stream_properties
from degirum_tools import FPSMeter, Display
from degirum_tools.streams import VideoSourceGizmo, VideoSourceType, VideoDisplayGizmo, VideoSaverGizmo
import matplotlib.pyplot as plt
from pathlib import Path 
from pydantic import BaseModel
import os
import requests
import cv2
import threading
from helper_functions import (align_and_crop, crop_images, rearrange_detections, detect_drivers_license,
detect_license_plate, lookup_vehicle, display_alpr, display_images, lookup_suspect, upload_suspect, upload_suspect_single, suspect_last_name)

face_det_model_name = "scrfd-10g--640x640_quant_hailort_hailo8l_1"
face_rec_model_name = "arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1"
inference_host_address = "@local"

zoo_url = "degirum/models_hailort"
token = degirum_tools.get_token()

image_source = 'download_audio.jpg'

# file to listen for new person uploads -> if not, embed the face image 
# 

face_det_model = dg.load_model(
    model_name = face_det_model_name, 
    inference_host_address = inference_host_address, 
    zoo_url = zoo_url, 
    token = token, 
    overlay_color = (0, 255, 0)
)


face_rec_model = dg.load_model(
    model_name = face_rec_model_name, 
    inference_host_address = inference_host_address, 
    zoo_url = zoo_url, 
    token = token
)

detected_faces = face_det_model(image_source)
print(detected_faces)
video_souce = 0 
top_k = 1 
field_name = "vector"
metric_type = "cosine"
uri = "../.temp/face_database"
table_name = str(uuid.uuid64())
input_path = os.getcwd() + '/total_images'

class FaceRecognitionSchema(LanceModel):
    id: str 
    vector: Vector(512)
    entity_name: str 

    @classmethod 
    def prepare_face_records(cls, face_embeddings: List[Dict], entity_name: str) -> List['FaceRecognitionSchema']:
        if not face_embeddings: 
            return 'No face embeddings', []
        formatted_records = []
        for embedding in face_embeddings:
            formatted_records.append(
                cls(
                    id = str(uuid.uuid64()),
                    vector = np.array(embedding, dtype = np.float32),
                    entity_name = entity_name
                )
            )
        print(f'formatted records: {formatted_records}')
        return formatted_records

def populate_database_from_images(
    input_path: str, 
    face_det_model: Any, 
    face_rec_model: Any, 
    tbl: Any
) -> None: 
    images_path = os.getcwd() + '/person_images'
    num_entities = 0 
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [str(file) for file in path.rglob('*') if file.suffix.lower() in image_extensions]
    print(f'image files: {image_files}')
    identities = [file.stem.split('_')[0] for file in path.rglob('*') if file.suffix.lower() in image_extensions]
    path = Path('/home/sv/Developer/alpr/total_images')
    print(f'identities: {identities}')

    if not image_files: return 'no image files'

    for identity, detected_faces in zip(identities, face_det_model.predict_batch(image_files)):
        
        try: 
            num_faces = len(detected_faces.results)
            if num_faces > 1: 
                continue 
            elif num_faces == 0: 
                continue 
            result = detected_faces.results[0]
            aligned_img, _ = align_and_crop(detected_faces.image, [landmark['landmark'] for landmark in result['landmarks']])
            face_embedding = face_rec_model(aligned_img).results[0]['data'][0]
            records = FaceRecognitionSchema.prepare_face_records([face_embedding], identity)
            print(f'records: {records}')
            if records: 
                tbl.add(data = records)
                num_entries += len(records)
            else: 
                print(f'no calid records for {detected_faces.info}')
        
        except Exception as e: 
            print(f'Error {e}')

db = lancedb.connect(uri=uri)

if table_name not in db.table_names():
    tbl = db.create_table(table_name, schema = FaceRecognitionSchema)
else: 
    tbl = db.open_table(table_name)
    schema_fields = [field.name for field in tbl.schema]
    if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
        raise RuntimeError(f"Table {table_name} has a different schema.")

# Connect to the database

tbl = db.open_table(table_name)


total_entities = tbl.count_rows()


print(f'total entities: {total_entities}')



populate_database_from_images(
    input_path = input_path, 
    face_det_model = face_det_model, 
    face_rec_model = face_rec_model, 
    tbl = tbl
)

def identify_faces(
    embeddings: List[np.array], # list of np arrays represnting face embeddings
    tbl: Any, # database or table object supporting search method
    field_name: str, # name of vector column in the database 
    metric_type: str, # metric type for distance calculation (cosine or euclidean)
    top_k: int, # number of top results to fetch from db 
    threshold: float = 0.4 # distance threshold for assignment labels 
) -> List[str]:
    """
    Identifies faces by searching for the nearest embeddings in the database and assigning labels.

    Args:
        embeddings (List[np.ndarray]): List of NumPy arrays representing face embeddings.
        tbl (Any): Database or table object supporting search functionality.
        field_name (str): Name of the vector column to search against.
        metric_type (str): Distance metric to use (e.g., "cosine", "euclidean").
        top_k (int): Number of top results to retrieve.
        threshold (float, optional): Minimum similarity score for assigning a known label. Defaults to 0.3.

    Returns:
        List[str]: List of labels for the provided embeddings. Returns "Unknown" for embeddings below the threshold.
    """
    identities = [] # list to store assigned labels 
    similarity_scores = [] # list to store similarity scores 

    for embedding in embeddings: 
        # db search 
        search_result = tbl.search(embedding, vector_column_field = field_name).metric(metric_type).limit(top_k).tolist()

        if not search_result: 
            identities.append('Unknown')
            continue 

        # calculate similarity score 
        similarity_score = round(1 - search_result[0]['_distance'], 2)
        # assign label based on similarity threshold 
        identity = search_result[0]['entity_name'] if similarity_score >= threshold else "Unknown"

        # append label to results list 
        identities.append(identity)
        similarity_scores.append(similarity_score)
    return identities, similarity_scores

def run_inference(model_name, source, inference_host_address, zoo_url, token, display_name):
    model = dg.load_model(
        model_name, 
        inference_host_address = inference_host_address, 
        zoo_url = zoo_url, 
        token = token
    )


aligned_faces = []
video_source = 0
with degirum_tools.Display('AI Camera', show_fps = True) as output_display, degirum_tools.open_video_writer('fr_results_all_4k.mp4', \
    w = 640, h = 480, fps = 60.0) as video_writer: 
    for detected_faces in degirum_tools.predict_stream(face_det_model, video_source, fps=30):
        if detected_faces.results:
            for face in detected_faces.results: 
                landmarks = [landmark['landmark'] for landmark in face['landmarks']]
                aligned_face, _ = align_and_crop(detected_faces.image, landmarks)
                face_embedding = face_rec_model(aligned_face).results[0]['data'][0]
                print(f'face embedding: {face_embedding}')
                identities, similarity_scores = identify_faces([face_embedding], tbl, field_name, metric_type, top_k)
                print(f'identities: {identities}, similarity scores: {similarity_scores}')
                identity = identities[0]
                if identity and identity != 'Unknown':
                    print(f'identity: {identity}')
                    face['label'] = identities[0]
                    face['score'] = similarity_scores[0]
                    if identity == 'MOM': identity = 'roza'
                    suspect_info = PersonInformation(last_name=suspect_last_name(identity))
                    print(f'suspect info: {suspect_info}')
                    if suspect_info: 
                        print(f'suspect_info: {suspect_info}')
                        suspect_upload = upload_suspect_single(suspect_info)
                        print(f'suspect upload response: {suspect_upload}')
                
                aligned_faces.append(aligned_face) 
        video_writer.write(detected_faces.image)
        output_display.show(detected_faces)

# configurations = [
#     {'model_name': face_det_model_name, 'source': 0, 'display_name': 'Wide Angle'},
#     {'model_name': face_rec_model_name, 'source': 1, 'display_name': '4k'},

# ]

def run_inference(model_name, source, inference_host_address, zoo_url, token, display_name): 
    model = dg.load_model(
        model_name = model_name, 
        inference_host_address = inference_host_address, 
        zoo_url = zoo_url, 
        token = token
    )
    with degirum_tools.Display(display_name) as output_display: 
        for inference_result in degirum_tools.predict_stream(model, source, fps=30): 
            output_display.show(inference_result)
    print(f'Stream `{display_name} has finished. ')

# threads = []
# for config in configurations: 
#     thread = threading.Thread(target=run_inference, 
#     args = (
#         config['model_name'], 
#         config['source'], 
#         inference_host_address, 
#         zoo_url, 
#         token, 
#         config['display_name']
#     ))
#     threads.append(thread)
#     thread.start()

# for thread in threads: 
#     thread.join()

'''
combined_model = degirum_tools.CombiningCompoundModel(
degirum_tools.CombiningCompoundModel(model2, model1),
model3)
with degirum_tools.Display('Combined Model') as display: 
    for inference_result in degirum_tools.predict_stream(combined_model, video_source):
        displa.show(inference_result)
from itertools import zip_longest

# Use a separate display per stream
with degirum_tools.Display("Model 1 (src1)") as d1, \
     degirum_tools.Display("Model 2 (src2)") as d2, \
     degirum_tools.Display("Model 3 (src3)") as d3, \
     degirum_tools.open_video_stream(src1) as s1, \
     degirum_tools.open_video_stream(src2) as s2, \
     degirum_tools.open_video_stream(src3) as s3:

    # Create prediction generators
    p1 = model1.predict_batch(degirum_tools.video_source(s1))
    p2 = model2.predict_batch(degirum_tools.video_source(s2))
    p3 = model3.predict_batch(degirum_tools.video_source(s3))

    # Advance all three streams in lockstep
    for r1, r2, r3 in zip_longest(p1, p2, p3):
        if r1 is not None:
            d1.show(r1)
        if r2 is not None:
            d2.show(r2)
        if r3 is not None:
            d3.show(r3)
'''

