from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import time
import json
import torch
import sys
import cv2
import os
from utils.general import check_img_size, check_imshow, increment_path
from utils.datasets import LoadImages, LoadWebcam
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from kafka import KafkaProducer

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# plt.style.use("dark_background")
# plt.style.use("seaborn-dark")
for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    plt.rcParams[param] = "0.9"  # very light grey

for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    plt.rcParams[param] = "#2E4F4F"  # bluish dark grey212946

from datetime import datetime
map_liqui = {"no detection":-1,"low": 1, "medium": 2, "high": 3}


class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)  # raw image signal
    yolo2main_res_img = Signal(np.ndarray)  # test result signal
    yolo2main_status_msg = Signal(
        str
    )  # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)  # fps
    yolo2main_liquidity = Signal(
        dict
    )  # Detected target results (number of each category)
    yolo2main_stream = Signal(
        dict
    )
    yolo2main_progress = Signal(int)  # Completeness
    # yolo2main_velocity = Signal(int)        # Number of categories detected
    yolo2main_target_num = Signal(int)  # Targets detected

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS["runs_dir"]) / self.args.task
        name = f"{self.args.mode}"
        self.save_dir = increment_path(
            Path(project) / name, exist_ok=self.args.exist_ok
        )
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None  # The detection model name to use
        self.new_model_name = "./models/v8_500.pt"  # Models that change in real time
        self.source = ""  # input source
        self.stop_dtc = False  # Termination detection
        self.continue_dtc = True  # pause
        self.save_res = False  # Save test results
        self.save_txt = False  # save label(txt) file
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar
        self.stream = False
        self.cust = "NA"
        self.projectid = "NA"
        self.batchid = 0
        self.flag =datetime.now() 

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.vid_cap = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info("")

            # set model
            self.yolo2main_status_msg.emit("Loading Model...")
            # print(self.new_model_name)
            # time.sleep(5)
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name
            stride = int(self.model.stride)  # model stride
            stride = 32
            imgsz = check_img_size(640, s=stride)  # check image size
            imgsz = 640
            # set source
            self.setup_source(
                self.source if self.source is not None else self.args.source
            )
            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            ):
                view_img = check_imshow()
                # cudnn.benchmark = True  # set True to speed up constant image size inference
                self.dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # print('aaaaaaaaaaaaaaaaa')
                # bs = len(dataset)  # batch_size
            else:
                self.dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            # # # Check save path/label

            if self.save_res or self.save_txt:
                (self.save_dir / "labels" if self.save_txt else self.save_dir).mkdir(
                    parents=True, exist_ok=True
                )

            # warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.pt or self.model.triton else self.dataset.bs,
                        3,
                        self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = (
                0,
                [],
                (ops.Profile(), ops.Profile(), ops.Profile()),
                None,
            )

            # start detection
            # for batch in self.dataset:

            self.percent_length = 1000
            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate
            batch = iter(self.dataset)

            while True:
                # Termination detection
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit("Detection terminated!")
                    self.vid_cap.release()
                    cv2.destroyAllWindows()
                    raise StopIteration
                    break
                # Change the model midway
                if self.used_model_name != self.new_model_name:
                    # self.yolo2main_status_msg.emit('Change Model...')
                    self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name
                # liquidity_values = '--'
                # if liquidity_values in ['medium','low']:
                #     self.setup_model('./models/modelv2.pt')
                #     self.used_model_name = './models/modelv2.pt'

                # pause switch
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit("Detecting...")
                    batch = next(self.dataset)  # next data

                    self.batch = batch
                    path, im, im0s, self.vid_cap = batch

                    visualize = (
                        increment_path(self.save_dir / Path(path).stem, mkdir=True)
                        if self.args.visualize
                        else False
                    )

                    # Calculation completion and frame rate (to be optimized)
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.yolo2main_fps.emit(str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(
                            count
                            / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            * self.percent_length
                        )
                        self.yolo2main_progress.emit(percent)
                    else:
                        percent = self.percent_length  # frame count +1
                    # if self.vid_cap:
                    # all_count = self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # total frames
                    # else:
                    #     all_count = 1
                    # self.progress_value = int(count/all_count*1000)         # progress bar(0~1000)
                    # if count % 5 == 0 and count >= 5:                     # Calculate the frame rate every 5 frames
                    #     self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                    #     start_time = time.time()

                    # preprocess

                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # inference

                    with self.dt[1]:
                        preds = self.model(
                            im, augment=self.args.augment, visualize=visualize
                        )
                    # postprocess

                    # self.yolo2main_status_msg.emit(str(a))

                    with self.dt[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    # visualize, save, write results
                    n = len(im)  # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            "preprocess": self.dt[0].dt * 1e3 / n,
                            "inference": self.dt[1].dt * 1e3 / n,
                            "postprocess": self.dt[2].dt * 1e3 / n,
                        }
                        # p, im0 = (path[i], im0s.copy()) if self.source_type.webcam or self.source_type.from_img \
                        #     else (path, im0s.copy())
                        im0 = im0s.copy()
                        p = path
                        p = Path(p)  # the source dir

                        # s:::   video 1/1 (6/6557) 'path':
                        # must, to get boxs\labels

                        label_str = self.write_results(
                            i, self.results, (p, im, im0)
                        )  # labels   /// original :s +=

                        # # # labels and nums dict
                        target_nums = 0
                        liquidity_values = "no detection"
                        self.labels_dict = {}
                        if "no detections" in label_str:
                            pass
                        else:
                            for ii in label_str.split(",")[:-1]:
                                nums, label_name = ii.split("~")
                                self.labels_dict[label_name] = int(nums)
                                # self.yolo2main_status_msg.emit(nums)
                                # time.sleep(10)
                                if label_name == "bubble":
                                    target_nums = int(nums)
                                # bubble_count =
                                if label_name in ["medium", "low", "high"]:
                                    liquidity_values = label_name

                        # # # save img or video result
                        if self.save_res:
                            self.save_preds(
                                self.vid_cap, i, str(self.save_dir / p.name)
                            )

                        # # # Send test results
                        self.yolo2main_res_img.emit(im0)  # after detection

                        self.yolo2main_pre_img.emit(
                            im0s if isinstance(im0s, np.ndarray) else im0s[0]
                        )  # Before testing
                        # self.yolo2main_liquidity.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_liquidity.emit(liquidity_values)
                        self.yolo2main_target_num.emit(target_nums)
                        # send to event Hub
                        
                        now =  datetime.now() 
                        try:
                            reading = {
                                "time":now,
                                "cust": self.cust,
                                "projectid": self.projectid,
                                "batchid": self.batchid,
                                "liquidity": map_liqui[liquidity_values],
                                "bubble_nums": target_nums,
                            }
                            self.yolo2main_stream.emit(reading)
                            # s = json.dumps(
                            #     reading
                            # )  # Convert the reading into a JSON string.
                            # with open("./realtime/data.json", "w") as f:
                            #     json.dump(s, f)

                        except KeyboardInterrupt:
                            # pass
                            print("Error")

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres / 1000)  # delay , ms

                    # self.yolo2main_progress.emit(self.progress_value)   # progress bar
                    if percent == self.percent_length:
                        self.yolo2main_progress.emit(0)
                        self.yolo2main_progress.emit("Detection completed")
                        # if hasattr(self, 'out'):
                        #     self.out.release()
                        break
                # Detection completed
                # if count + 1 >= all_count:
                #     if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                #         self.vid_writer[-1].release()  # release final video writer
                #     self.yolo2main_progress.emit('Detection completed')
                #     break

        except Exception as e:
            pass
            print(e)
            self.yolo2main_status_msg.emit("%s" % e)

    def get_annotator(self, img):
        return Annotator(
            img, line_width=self.args.line_thickness, example=str(self.model.names)
        )

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        ### important
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            # Tích hợp DeepSORT ở đây
            # Chuyển đổi `pred` thành định dạng mong muốn của DeepSORT
            # Ví dụ: [x1, y1, x2, y2, confidence, class]
            # deepsort_input = pred[:, :5].cpu().numpy()
            # deepsort_input[:, 4] = deepsort_input[:, 4] * pred[:, 5].cpu().numpy()  # Nhân confidence với class score
            # tracks = deepsort.update(deepsort_input)

            # `tracks` là danh sách các đối tượng được theo dõi, mỗi đối tượng có thông tin [x1, y1, x2, y2, track_id]
            # Bạn có thể sử dụng thông tin này để vẽ bounding box và ID của đối tượng

            results.append(
                Results(
                    orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred
                )
            )  # , tracks=tracks))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if (
            self.source_type.webcam or self.source_type.from_img
        ):  # batch_size >= 1         # attention
            log_string += f"{idx}: "
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, "frame", 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / "labels" / p.stem) + (
            "" if self.dataset.mode == "image" else f"_{frame}"
        )
        # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f"{log_string}(no detections), "  # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class

            # _ , n = n.split(':')
            log_string += f"{n}~{self.model.names[int(c)]},"  #   {'s' * (n > 1)}, "   # don't add 's'
            # now log_string is the classes 👆
            if ":" in log_string:
                _, log_string = log_string.split(":")
            # self.yolo2main_status_msg.emit(str(log_string))
            # time.sleep(100)

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.save_txt:  # Write to file
                line = (
                    (cls, *(d.xywhn.view(-1).tolist()), conf)
                    if self.args.save_conf
                    else (cls, *(d.xywhn.view(-1).tolist()))
                )  # label format
                with open(f"{self.txt_path}.txt", "a") as f:
                    f.write(("%g " * len(line)).rstrip() % line + "\n")
            if (
                self.save_res or self.args.save_crop or self.args.show or True
            ):  # Add bbox to image(must)
                # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

                # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa',c)
                # print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',d.id.item())

                # self.yolo2main_progress.emit(c)
                # time.sleep(1000)
                c = int(cls)  # integer class
                name = (
                    f"id:{int(d.id.item())} {self.model.names[c]}"
                    if d.id is not None
                    else self.model.names[c]
                )
                label = (
                    None
                    if self.args.hide_labels
                    else (name if self.args.hide_conf else f"{name} {conf:.2f}")
                )
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(
                    d.xyxy,
                    imc,
                    file=self.save_dir
                    / "crops"
                    / self.model.model.names[c]
                    / f"{self.data_path.stem}.jpg",
                    BGR=True,
                )

        return log_string





class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot()
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = (
        Signal()
    )  # The main window sends an execution signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(
            Qt.FramelessWindowHint
        )  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        # UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        # UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        # UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        # UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        bootstrap_servers = "localhost:9093"
        self.topic = "states"  # Kafka topic to send data

        # Create Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            # Use default serialization
            value_serializer=lambda v: v,
            api_version=(2, 0, 2),
        )
        # read model folder
        self.pt_list = os.listdir("./models")
        self.pt_list = [file for file in self.pt_list if file.endswith(".pt")]
        # self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(
            self
        )  # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread

        self.yolo_predict = YoloPredictor()  # Create a Yolo instance
        self.select_model = self.model_box.currentText()
        # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.yolo_thread = QThread()  # Create yolo thread
        self.yolo_predict.yolo2main_pre_img.connect(
            lambda x: self.show_image(x, self.pre_video)
        )
        self.yolo_predict.yolo2main_res_img.connect(
            lambda x: self.show_image(x, self.res_video)
        )
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_liquidity.connect(lambda x:self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_liquidity.connect(
            lambda x: self.Class_num.setText(str(x))
        )
        

        self.yolo_predict.yolo2main_stream.connect(
                lambda x: self.stream_data(x)
            )

        self.yolo_predict.yolo2main_target_num.connect(
            lambda x: self.Target_num.setText(str(x))
        )
        self.yolo_predict.yolo2main_target_num.connect(
            lambda x: self.update_plot_bubble(x)
        )

        self.yolo_predict.yolo2main_progress.connect(
            lambda x: self.progress_bar.setValue(x)
        )
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "iou_spinbox")
        )  # iou box
        self.iou_slider.valueChanged.connect(
            lambda x: self.change_val(x, "iou_slider")
        )  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "conf_spinbox")
        )  # conf box
        self.conf_slider.valueChanged.connect(
            lambda x: self.change_val(x, "conf_slider")
        )  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "speed_spinbox")
        )  # speed box
        self.speed_slider.valueChanged.connect(
            lambda x: self.change_val(x, "speed_slider")
        )  # speed scroll bar
        # Prompt window initialization
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        # self.Liquidity_type.setText('--')

        self.fps_label.setText("--")
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)  # chose_cam
        self.src_rtsp_button.clicked.connect(
            self.show_status("The function has not yet been implemented.")
        )  # chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination
        # self.stream_button.toggled.connect(self.stream_data)
        # Other function buttons
        # self.save_res_button.toggled.connect(self.update_plot_bubble)  # save image option

        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        # self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  # top right settings button
        # Create a new QVBoxLayout to hold the chart
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.verticalLayout_17.addWidget(self.canvas)

        # self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        # self.des.setLayout(QVBoxLayout())
        # self.des.layout().addWidget(self.canvas2)
        n_data = 50
        # date = [datetime.datetime.now() + datetime.timedelta(hours=i) for i in range(n_data)]
        self.xdata = [datetime.now()] * n_data
        # self.xdata  = matplotlib.dates.date2num(date)
        self.ydata = [0 for i in range(n_data)]
        # self.ydata2= [0 for i in range(n_data)]
        # self.ydata3= [0 for i in range(n_data)]
        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888,
            )
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def getlabel(self, msg):
        return msg

    # Control start/pause
    
    def run_or_continue(self):
        if self.yolo_predict.source == "":
            self.show_status(
                "Please select a video source before starting detection..."
            )
            self.run_button.setChecked(False)

        elif (
            self.lineEdit_cust.text() == ""
            or self.lineEdit_project.text() == ""
            or self.lineEdit_batchid.text() == ""
        ):
            self.show_status(
                "Please input user information before starting detection..."
            )
            self.run_button.setChecked(False)
        elif not self.lineEdit_batchid.text().isnumeric():
            self.show_status("BatchID must be numberic...")
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False

            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # start button
                self.save_txt_button.setEnabled(
                    False
                )  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                # self.stream_button.setEnabled(False)

                self.yolo_predict.cust = self.lineEdit_cust.text()
                self.yolo_predict.projectid = self.lineEdit_project.text()
                self.yolo_predict.batchid = self.lineEdit_batchid.text()

                self.show_status("Detecting...")
                self.yolo_predict.continue_dtc = True  # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)  # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == "Detection completed":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.stream_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # end process
        elif msg == "Detection terminated!":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            # self.stream_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # end process
            self.pre_video.clear()  # clear image display
            self.res_video.clear()
            self.yolo_predict.vid_cap.release()
            cv2.destroyAllWindows()
            self.Class_num.setText("--")
            self.Target_num.setText("--")
            self.fps_label.setText("--")

    def update_plot_bubble(self, msg1):
        if self.visual_button.isChecked():
            # Drop off the first y element, append a new one.
            self.xdata = self.xdata[1:] + [datetime.now()]

            self.ydata = self.ydata[1:] + [msg1]

            self.canvas.axes.cla()  # Clear the canvas.

            self.canvas.axes.plot(
                self.xdata, self.ydata, color="#08F7FE", label="Bubble"
            )

            # Trigger the canvas to update and redraw.
            self.canvas.axes.legend(loc="upper right", bbox_to_anchor=(-0.04, 1))

            self.canvas.draw()

    def visualize_result(self, msg):
        # if self.saveCheckBox2.isChecked():

        try:
            self.update_plot(msg)
            self.show()

            # Setup a timer to trigger the redraw by calling update_plot.
            self.timer = QTimer()
            self.timer.setInterval(1000000)
            self.timer.timeout.connect(self.update_plot)
            self.timer.start()

        except Exception as e:
            print(repr(e))
    
    def stream_data(self,msg):
        if self.stream_button.isChecked():
            # try:
                self.yolo_predict.stream = True
                
                time_difference = (msg["time"]-self.yolo_predict.flag).total_seconds()
               



                if abs(time_difference)>5:
                    self.yolo_predict.flag =  datetime.now()
                    msg["time"]=msg["time"].strftime("%m/%d/%Y, %H:%M:%S")
                    print(msg)
                    self.producer.send(self.topic, value=str(msg).encode("utf-8"))
                    self.producer.flush()
                else:
                    pass

            

            # except Exception as e:
                # self.show_status("%s" % e)
                # pass
           
    # else:
    #     if hasattr(self, 'timer') and self.timer.isActive():
    #         self.timer.stop()
    # self.hide()

    # select local file
    def open_src_file(self):
        config_file = "config/fold.json"
        config = json.load(open(config_file, "r", encoding="utf-8"))
        open_fold = config["open_fold"]
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self,
            "Video/image",
            open_fold,
            "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)",
        )
        if name:
            self.yolo_predict.source = name
            self.show_status("Load File：{}".format(os.path.basename(name)))
            config["open_fold"] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_json)
            self.stop()

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button,
                title="Note",
                text="loading camera...",
                time=2000,
                auto=True,
            ).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet("""
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            """)

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()
                self.show_status("Loading camera：{}".format(action.text()))

        except Exception as e:
            self.show_status("%s" % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = "config/ip.json"
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, "r", encoding="utf-8"))
            ip = config["ip"]
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(
            lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text())
        )

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button,
                title="Load rtsp",
                text="Nhap rtsp...",
                time=1000,
                auto=True,
            ).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open("config/ip.json", "w", encoding="utf-8") as f:
                f.write(new_json)
            self.show_status("Loading rtsp：{}".format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status("%s" % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Run image results are not saved.")
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Run image results will be saved.")
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Labels results are not saved.")
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Labels results will be saved.")
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = "config/setting.json"
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {
                "iou": iou,
                "conf": conf,
                "rate": rate,
                "save_res": save_res,
                "save_txt": save_txt,
            }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, "r", encoding="utf-8"))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config["iou"]
                conf = config["conf"]
                rate = config["rate"]
                save_res = config["save_res"]
                save_txt = config["save_txt"]
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = False if save_res == 0 else True
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = False if save_txt == 0 else True
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # end thread
        self.yolo_predict.continue_dtc = False

        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        self.save_txt_button.setEnabled(True)  # Ability to use the save button
        # self.stream_button.setEnabled(True)  # Ability to use the save button

        self.pre_video.clear()  # clear image display
        self.res_video.clear()  # clear image display
        if self.yolo_predict.vid_cap is not None:
            self.yolo_predict.vid_cap.release()
            cv2.destroyAllWindows()
        self.progress_bar.setValue(0)
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        self.fps_label.setText("--")

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == "iou_spinbox":
            self.iou_slider.setValue(
                int(x * 100)
            )  # The box value changes, changing the slider
        elif flag == "iou_slider":
            self.iou_spinbox.setValue(
                x / 100
            )  # The slider value changes, changing the box
            self.show_status("IOU Threshold: %s" % str(x / 100))
            self.yolo_predict.iou_thres = x / 100
        elif flag == "conf_spinbox":
            self.conf_slider.setValue(int(x * 100))
        elif flag == "conf_slider":
            self.conf_spinbox.setValue(x / 100)
            self.show_status("Conf Threshold: %s" % str(x / 100))
            self.yolo_predict.conf_thres = x / 100
        elif flag == "speed_spinbox":
            self.speed_slider.setValue(x)
        elif flag == "speed_slider":
            self.speed_spinbox.setValue(x)
            self.show_status("Delay: %s ms" % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()

        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status("Change Model：%s" % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir("./models")
        pt_list = [file for file in pt_list if file.endswith(".pt")]
        # pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

        cust_id = self.lineEdit_cust.text()
        if cust_id != self.yolo_predict.cust:
            self.yolo_predict.cust = cust_id
        project_id = self.lineEdit_project.text()
        if project_id != self.yolo_predict.projectid:
            self.yolo_predict.projectid = project_id
        batch_id = self.lineEdit_batchid.text()
        if batch_id != self.yolo_predict.batchid:
            self.yolo_predict.batchid = batch_id

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):

        config_file = "config/setting.json"
        config = dict()
        config["iou"] = self.iou_spinbox.value()
        config["conf"] = self.conf_spinbox.value()
        config["rate"] = self.speed_spinbox.value()
        config["save_res"] = (
            0 if self.save_res_button.checkState() == Qt.Unchecked else 2
        )
        config["save_txt"] = (
            0 if self.save_txt_button.checkState() == Qt.Unchecked else 2
        )
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_json)
        self.producer.close()
        
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            self.producer.close()

            MessageBox(
                self.close_button,
                title="Note",
                text="Exiting, please wait...",
                time=3000,
                auto=True,
            ).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
