import os
import shutil as sh
import multiprocessing as mu
from sys import executable
from glob import glob
from yaml import safe_load
from importlib.util import find_spec

require = "torch torchvision torchaudio", "ultralytics hyperlpr3 PyQt5"
url = "mirrors.nju.edu.cn/pytorch/whl/cu129", "pypi.tuna.tsinghua.edu.cn/simple"
for require, url in zip(require, url):
    if not all(map(find_spec, require.split())):  # 下载缺失库
        os.system(f"{executable} -m pip install {require} -i https://{url}")
os.system(
    f'{executable} -c "import hyperlpr3" && {"cls" if "nt" == os.name else "clear"}'
)
for file in glob(os.path.expanduser("~/.hyperlpr3/*.zip")):
    os.remove(file)
import cv2
import numpy as np
from torch.cuda import is_available
from hyperlpr3 import LicensePlateCatcher
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def getinfo(source):
    # source:int|str,摄像头编号/视频文件路径
    cap = cv2.VideoCapture(source)
    width, height, fps = (round(cap.get(x)) for x in range(3, 6))
    cap.release()
    return fps, height, width


class model(YOLO):
    detect = LicensePlateCatcher(detect_level=1)  # 高精度

    # 设置模型
    def set(self, model=""):
        # model:str,模型yaml/pt文件路径,默认'',最新
        self.__dict__ = {"project": self.project}
        for model in (
            model or f"runs/detect/{self.project}/train/weights/best.pt",
            f"result/{self.project}_best.pt",
        ):
            if os.path.exists(model):
                super().__init__(model)
                break

    def __init__(self, project, model=""):
        # project:str,项目名
        self.project = project
        self.set(model)

    def __del__(self):
        file = f"runs/detect/{self.project}/train/weights/best.pt"
        if os.path.exists(file):
            os.makedirs("result", exist_ok=True)
            sh.copy2(file, f"result/{self.project}_best.pt")
        sh.rmtree("runs/detect/" + self.project, True)
        for file in "runs/detect runs result".split():
            if os.path.exists(file) and not os.listdir(file):
                os.rmdir(file)

    # 训练模型
    @classmethod
    def train(cls, data, epochs=100, batch=16, model="yolo26s.pt"):
        """
        data:str,数据集yaml文件路径
        epochs:int,轮数,默认100
        batch:int,批大小,默认16
        """
        source = safe_load(open(data, encoding="utf-8"))
        source = os.path.join(source["path"], source["train"])
        imgsz = []
        for file in os.listdir(source):
            file = cv2.imread(os.path.join(source, file))
            if not file is None:
                imgsz.append(max(file.shape[:2]))  # 较长边
        imgsz = file = max(704, int(np.percentile(imgsz, 70)))  # 70%位数
        project = os.path.splitext(os.path.basename(data))[0]
        cls.__base__(model).train(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            patience=max(1, epochs // 10),
            exist_ok=True,
            device="all" if is_available() else "cpu",
            optimizer="SGD",  # 随机梯度下降,泛化强
        )
        return cls(project)

    # 生产者
    def producer(self, queue: mu.Queue, source, num, rate, imgsz, conf, iou):
        """
        queue:通信管道
        num:int,停留帧数阈值
        rate:float,位移归一化阈值
        imgsz:tuple[int],图片高宽元组
        conf:float,置信度阈值
        iou:float,交并比阈值
        """
        last = {}
        for i, res in enumerate(
            super().track(
                source,
                True,
                True,
                imgsz=imgsz,  # (高,宽)
                conf=conf,
                iou=iou,
                device=-1,  # 最空闲GPU
                verbose=False,
            )
        ):
            img, res = res.orig_img, res.boxes
            plot = Annotator(img, 2, 25, "SourceHanSansSC-VF.ttf", True)
            if not res.id is None:
                for id, cls, pos, xyxy in zip(res.id, res.cls, res.xywhn, res.xyxy):
                    id, pos = int(id), pos[:2].cpu().numpy()
                    car = last.get(id, {})
                    if not (0 < len(car.get("plate", "")) < 9 and i % 50):  # 车牌更新
                        x1, y1, x2, y2 = map(round, xyxy.cpu().numpy())
                        try:  # 车牌识别
                            car["plate"] = self.detect(img[y1:y2, x1:x2])[0][0]
                            car["plate"] = f"{car['plate'][:2]}·{car['plate'][2:]}"
                        except:
                            car["plate"] = ""
                    park = (np.linalg.norm(pos - car.get("pos", 3)) < rate) * (
                        1 + car.get("park", 0)
                    )  # 新停车时间
                    last[id] = car = {
                        "cls": super().names[int(cls)],
                        "plate": car["plate"],
                        "illegal": num < park,
                        "park": park,
                        "pos": pos,
                        "time": i,
                    }
                    plot.box_label(
                        xyxy,
                        f"{car['plate'] or f'id:{id}'} {car['cls']}",
                        (0, 255 * (park < num), 255 * bool(park)),  # BGR数组
                        (0, 0, 0),
                    )
            if not i % 100:
                for id, car in list(last.items()):
                    if car["time"] + 99 < i:  # 离开图像
                        last.pop(id)
            queue.put((np.array(plot.im).astype(np.uint8), last.copy()))  # 浅复制
        queue.put(None)  # 完成信号

    # 消费者
    def track(
        self,
        source=0,
        time=5,  # float,停留秒数阈值
        rate=3e-3,
        show=lambda x: 1 + cv2.waitKey(cv2.imshow("track", x) or 16),  # 展示函数
        recv=lambda x: [print(a, b) for a, b in x.items() if b["illegal"]],  # 接收函数
        conf=0.25,
        iou=0.7,
        save=False,  # bool,是否保存结果,默认False
    ):
        info = getinfo(source)
        queue = mu.Queue(2 * info[0])
        produce = mu.Process(
            target=self.producer,
            args=(queue, source, time * info[0], rate, info[1:], conf, iou),
        )
        produce.start()
        if save:
            os.makedirs("result", exist_ok=True)
            save = cv2.VideoWriter(
                f"result/{os.path.basename(str(source))}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                info[0],
                info[:0:-1],  # (宽,高)
            )
        while True:
            data = queue.get()
            if data == None:
                break
            else:
                recv(data[1])
                if show(data[0]):
                    queue.put(produce.terminate())  # 结束生产
                if save:
                    save.write(data[0])
        produce.join()
        if save:
            save.release()
        cv2.destroyAllWindows()
