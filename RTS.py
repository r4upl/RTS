from gpiozero import Robot, DistanceSensor, OutputDevice
from time import sleep
import time, sys
from datetime import datetime, timedelta

from threading import Thread
from multiprocessing import Process, Manager, cpu_count
import threading
import click
#import keyboard
import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import picamera

import smtplib                    
import os                        

# Добавляем необходимые подклассы - MIME-типы
# Импорт класса для обработки неизвестных MIME-типов, базирующихся на расширении файла 
import mimetypes                                            
from email import encoders                                  
from email.mime.base import MIMEBase                        
from email.mime.text import MIMEText                        
from email.mime.image import MIMEImage                      
from email.mime.multipart import MIMEMultipart              

n = 0
# глобальные переменные
known_face_encodings = []
known_face_metadata = []
global_stop_trigger = False
running = True
change_direction = False



# Получить следующий номер процесса
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Получить предыдущий номер процесса
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1

# Функция сохранения распознанных лиц в файл на диск
def save_known_faces(Global_known_face_encodings, Global_known_face_metadata):
    with open("faces_list.dat", "wb") as face_data_file:
        face_data = [[*Global_known_face_encodings], [*Global_known_face_metadata]]
        pickle.dump(face_data, face_data_file)
        print("Faces save to disk.",Global_known_face_encodings)

# Функция регистрации новых распознанных лиц, определения их легальности и отправки уведомления на электронную почту в случае нелегальности
def register_new_face(face_encoding, face_image, Global_known_face_encodings, Global_known_face_metadata, Global, frame_process):
# Определение легальности лица - легальные лица записываются в течение заданного времени 
# (60 секунд, не учитывая время загрузки ПО РТС) после старта ПО РТС, все остальные 
# в дальнейшем обнаруженные РТС лица считаются нелегальными    
    (Global_known_face_encodings).append(face_encoding)
    i = len(Global_known_face_encodings)-1
    i = int(i)
    a=Global.timeToSave - time.time()
    if a>0:
        Global.isLegal = 1
    else:
        Global.isLegal = 0

    if Global.isLegal:
        (Global_known_face_metadata).append({
            "first_detection": datetime.now(),
            "first_detection_this_interaction": datetime.now(),
            "last_detection": datetime.now(),
            "detection_count": 1,
            "detection_frames": 1,
            "isLegal": 1,
            "isSendToMail":1,
            "face_image": face_image,
        })
    else:
        (Global_known_face_metadata).append({
            "first_detection": datetime.now(),
            "first_detection_this_interaction": datetime.now(),
            "last_detection": datetime.now(),
            "detection_count": 1,
            "detection_frames": 1,
            "isLegal": 0,
            "isSendToMail":0,
            "face_image": face_image,
        })

    print("new face register",Global_known_face_encodings)
    # Отправить сообщение с информацией о нарушителе по почте

    if (Global_known_face_metadata[i]["isLegal"] == 0) and (Global_known_face_metadata[i]["isSendToMail"] == 0):
        addr_to   = "example_mail@mail.ru"                                # Почтовый адрес получателя
        msg = []
        msg.append({
            "info":"Информация о незарегистрированном посетителе:",
            "first_detection":Global_known_face_metadata[i]["first_detection"],
            "first_detection_this_interaction":Global_known_face_metadata[i]["first_detection_this_interaction"],
            "last_detection":Global_known_face_metadata[i]["last_detection"],
            "detection_count":Global_known_face_metadata[i]["detection_count"],
            "detection_frames":Global_known_face_metadata[i]["detection_frames"]
        })

        msg = " ".join(map(str,msg))
        send_to_mail(addr_to, "Обнаружен незарегистрированный посетитель", msg, frame_process)
        Global_known_face_metadata[i]["isSendToMail"] = 1

# Функция отправки сообщения на электронную почту, содержащего фото предполагаемого нарушителя и сведения о времени обнаружения
def send_to_mail(addr_to, msg_subj, msg_text, face_image):

    addr_from = "mail_from_robot@mail.ru"                   # Отправитель
    password  = "strong_password"                           # Пароль для внешнего приложения

    msg = MIMEMultipart()                                   # Формируем сообщение
    msg['From']    = addr_from                              # Указываем адресата
    msg['To']      = addr_to                                # Указываем получателя
    msg['Subject'] = msg_subj                               # Указываем тему сообщения
    body = msg_text                                         # Указываем текст сообщения
    msg.attach(MIMEText(body, 'plain'))                     # Добавляем в сообщение текст
    
    is_success, im_buf = cv2.imencode(".png", face_image)
    # Конвертирование изображения в массив numpy
    data_encode = np.array(im_buf)
  
    # Конвертирование массива numpy в байты
    byte_encode = data_encode.tobytes()
    # прикрепление изображения к письму
    msg.attach(MIMEImage(byte_encode))

    #======== Настройка почтового провайдера mail.ru  ===============================================
    server = smtplib.SMTP_SSL('smtp.mail.ru', 465)          # Создаем объект SMTP
    server.login(addr_from, password)                       # Доступ
    server.send_message(msg)                                # Отправляем сообщение
    server.quit()                                           # Выходим
    #================================================================================================

# Функция поиска ранее распознанных лиц
def lookup_known_face(face_encoding, Global_known_face_encodings, Global_known_face_metadata):

    metadata = None

   
    if len(Global_known_face_encodings) == 0:
        print("Global_known_face_encodings = 0")
        return metadata


    face_distances = face_recognition.face_distance(Global_known_face_encodings, face_encoding)

    index_best_match = np.argmin(face_distances)

    if face_distances[index_best_match] < 0.65:
       
        metadata = Global_known_face_metadata[index_best_match]
       
        metadata["last_detection"] = datetime.now()
        metadata["detection_frames"] += 1

        if datetime.now() - metadata["first_detection_this_interaction"] > timedelta(minutes=5):
            metadata["first_detection_this_interaction"] = datetime.now()
            metadata["detection_count"] += 1

    else:      
        print("noop")

    return metadata

# Поток для определения препятствий
def wall_watch():
    global global_stop_trigger
    global change_direction
    while not global_stop_trigger:
        valueL = sensorL.distance
        valueR = sensorR.distance
        sleep(0.000000000000000001)
        if (valueL < 0.20) or (valueR < 0.20): # срабатывание на расстоянии 20 см
            change_direction = True


# Поток для управления движением робота
def move_robot():
    global global_stop_trigger
    global change_direction
    while not global_stop_trigger:

        if not change_direction:
            robot.forward(0.3)
            sleep(0.5)
        
        else:
            robot.right(0.3)
            sleep(2)
            robot.forward(0.3)
            sleep(0.5)
            change_direction = False

# Поток получения кадра с камеры
def capture(read_frame_list, Global, worker_num):
    video_capture = picamera.PiCamera()
    video_capture.resolution = (1312,736)
    # Выделение области интереса (ROI)
    video_capture.zoom = (0.25,0.25,0.5,0.5)
    # Автоконтраст
    video_capture.awb_mode = 'auto'
    video_capture.framerate = 15
    # резкость от -100 до 100
    video_capture.sharpness = 50 
    video_capture.video_stabilization = True
    time.sleep(2)
    output = np.empty((368, 672, 3), dtype=np.uint8)
   
    while not Global.is_exit:
        # Чтение кадра
        if Global.buff_num != next_id(Global.read_num, worker_num):
            frame =  video_capture.capture(output, format = 'bgr', resize=(672,368))
            frame = output.reshape((368,672,3))
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    video_capture.close()

# Процессы, выполняющие параллельную обработку алгоритма распознавания лиц. Для 4х ядер ЦП создается 3 параллельных процесса 
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num, Global_known_face_encodings, Global_known_face_metadata):

    while not Global.is_exit:

        # Ожидание чтения
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # Если завершается приложения, прекращается ожидание кадров от камеры
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Чтение одного кадра из списка кадров
        frame_process = read_frame_list[worker_id]

        # Передача очереди чтения кадра следующему процессу
        Global.read_num = next_id(Global.read_num, worker_num)

        # Конвертация изображения из цвета BGR (который использует OpenCV) в цвет RGB (который использует face_recognition)
        rgb_frame = np.ascontiguousarray(frame_process[:, :, ::-1])
    
        # Поиск всех лиц и 128-размерных векторов лиц в кадре от камеры, занимает больше всего времени ЦП. Выбираем алгоритм HoG
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):

            metadata = lookup_known_face(face_encoding, Global_known_face_encodings, Global_known_face_metadata)

            if metadata is not None:
                time_at_camera = datetime.now() - metadata['first_detection_this_interaction']

                if metadata["isLegal"] == 1:
                    face_label = f"Legal, At camera {int(time_at_camera.total_seconds())}s "
                else:
                    face_label = f"NOT LEGAL! At camera {int(time_at_camera.total_seconds())}s "


            else:
                face_label = "New person!"
                top, right, bottom, left = face_location
                face_image = frame_process[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))
                cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
                register_new_face(face_encoding, face_image, Global_known_face_encodings, Global_known_face_metadata, Global, frame_process)

            face_labels.append(face_label)

        # Просмотр каждого лица в кадре
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):

            if face_label[0:5] == "Legal":
                cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame_process, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame_process, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        ########
        Global.number_of_recent_persons = 0
        for metadata in Global_known_face_metadata:
            if datetime.now() - metadata["last_detection"] < timedelta(seconds=10) and metadata["detection_frames"] > 5:
                x_position = Global.number_of_recent_persons * 150
                frame_process[30:180, x_position:x_position + 150] = metadata["face_image"]
                Global.number_of_recent_persons += 1

                visits = metadata['detection_count']
                visit_label = f"{visits} visits"
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame_process, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if Global.number_of_recent_persons > 0:
            cv2.putText(frame_process, "persons at camera", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        ########

        if len(face_locations) > 0 and Global.number_of_faces_since_save > 100:
            save_known_faces(Global_known_face_encodings, Global_known_face_metadata)
            Global.number_of_faces_since_save = 0
        else:
            Global.number_of_faces_since_save += 1

        # Ожидание для отображения кадра на экране
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Отправить кадр в глобальных список для отображения на экране
        write_frame_list[worker_id] = frame_process

        # Передача очереди отображения кадра на экране следующему процессу
        Global.write_num = next_id(Global.write_num, worker_num)    


if __name__ == "__main__":
    # Инициализация входных и выходных устройств
    sensorL = DistanceSensor(echo = 17, trigger = 4)
    sensorR = DistanceSensor(echo = 9, trigger = 10)
    robot = Robot(left=(25, 24), right=(23, 18))

    # Запустить поток для определения препятствий
    t1 = Thread(target=wall_watch)
    t1.start()

    # Запустить поток для управления движением робота
    t2 = Thread(target=move_robot)
    t2.start() 

    # Глобальные переменные
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    # Метка легальных субъектов. Запись легальных субъектов производится в течение 60 секунд после запуска ПО
    Global.isLegal = 1
    Global.timeToSave = time.time() + 60
    Global_known_face_encodings = Manager().list()
    Global_known_face_metadata = Manager().list()
    Global.number_of_recent_persons = 0
    Global.number_of_faces_since_save = 0
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Определение числа процессов, обрабатывающих алгоритм распознавания лиц, для 4х ядер ЦП создается 3 процесса
    if cpu_count() > 2:
        worker_num = cpu_count() - 1 
    else:
        worker_num = 2
    print("cpu_count", cpu_count())

    #загружаем известные лица из файла
    try:
        with open("faces_list.dat", "rb") as face_data_file:
            [[*Global_known_face_encodings], [*Global_known_face_metadata]] = pickle.load(face_data_file)
            print("Known faces loaded from disk.",Global_known_face_encodings)
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass
    
    # Список процессов
    p = []

    # Создание потока для чтения кадров от камеры
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

    # Создание процессов
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num, Global_known_face_encodings, Global_known_face_metadata,)))
        p[worker_id].start()
        
    # Запуск отображения на экране видео от камеры (в случае использования робота без необходимости просмотра изображения можно удалить данную часть кода)
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Примерное вычисление FPS 
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / np.sum(fps_list)
            print("fps: %.2f" % fps)

            n = n+1    
            print("find", n)
            # Отображение на экране изображения от камеры с наложенными метками после распознавания лиц
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])


        # Ожидание нажатия клавиши "q" для выхода из ПО РТС
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Сохранение известных лиц в файл
            save_known_faces(Global_known_face_encodings, Global_known_face_metadata)
            Global.is_exit = True
            global_stop_trigger = True
            break
        time.sleep(0.01)


    # Уничтожение окон
    cv2.destroyAllWindows()

    # Остановка движения робота   
    robot.stop()

