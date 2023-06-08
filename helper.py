from ultralytics import YOLO
import streamlit as st
import cv2

import settings


def load_model(model_path):
    """
    Загружает модель обнаружения объектов YOLO из указанного пути model_path.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Отображение обнаруженных объектов на видеокадре.

    Args:
    - conf: Порог доверия для обнаружения объектов.
    - model (YoloV8): Модель обнаружения объектов YOLOv8.
    - st_frame (объект Streamlit): Объект Streamlit для отображения обнаруженного видео.
    - image: Изображение.
    - is_display_tracking (bool): Флаг, указывающий, следует ли отображать отслеживание объектов (по умолчанию = Нет)
    """

    # Изменение размера изображения до стандартного размера
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Отображение отслеживания объектов, если true
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Прогнозирование объектов на изображении с помощью модели YOLOv8
        res = model.predict(image, conf=conf)

    # Отобразить обнаруженные объекты на видеокадр
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_webcam(conf, model):
    """
    Воспроизводит поток с веб-камеры. Обнаруживает объекты в реальном времени.
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Воспроизводит сохраненный видеофайл. Отслеживает и обнаруживает объекты в режиме реального времени.
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
