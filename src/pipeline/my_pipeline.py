import os

import torch
from lightning.fabric import Fabric
from lightning_fabric.loggers import TensorBoardLogger
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
from tqdm import tqdm

from src.utils import get_logger

log = get_logger(__name__)


class MyPipeline:
    """
    所有的训练步骤
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        TensorBoardLog: TensorBoardLogger,
        net_input_size: list,
        num_classes: int,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: int = 1,
        precision: str = "32",
        matmul_precision: str = "high",
    ):
        """

        :param net: torch.nn.Module类型的网络结构;
        :param net_input_size: net输入大小，用于保存在tb上;
        :param loss: 如torch.nn.CrossEntropyLoss()类型的损失;
        :param optimizer: torch.optim.Optimizer类型的优化器;
        :param TensorBoardLog: TensorBoardLogger类型的日志;

        :param num_classes: 网络分类数;
        :param accelerator: Fabric加速器类型:默认自动选择."cpu","cuda", "mps", "gpu", "tpu", "auto".
        :param strategy: Fabric加速器策略;默认为自动选择."single_device", "dp", "ddp", "ddp_spawn", "deepspeed", "ddp_sharded".
        :param devices: Fabric加速器设备数:默认为自动选择.list[int] 代表指定索引设备运行,.int 代表指定多个设备.-1 代表所有设备一起运行.
        :param precision: Fabric加速器精度： 默认以32位精度运行;
            "32": 32位精度（模型权重保留在torch.float32中).
            "16-mixed"：(16位混合精度（模型重量保留在torc.float32中).
            "bf16-mixed":(16位bfloat混合精度（模型重量保留在torc.float32中).
            ”16_true": 16位精度(模型权重转换为 torch.float16).
            "bf16-true":16位 bfloat 精度(模型权重转换到 torch.bfloat16).
            "64":64位(双)精度(模型权重转换为 torch.float64).
            "transformer-engine":通过 TransformerEngine (Hopper GPU 和更高版本)实现8位混合精度.
            详情： https://lightning.ai/docs/fabric/stable/fundamentals/precision.html.

        """
        super().__init__()
        # 初始化Fabric，用于切换多设备/分布式/混合精度
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
        )
        torch.set_float32_matmul_precision(matmul_precision)

        self.num_classes = num_classes
        # 初始化网络
        # with self.fabric.init_module():
        self.net = net
        # 损失函数
        self.loss = loss
        # 优化器
        self.optim = optimizer(params=self.net.parameters())
        # 在初始化中添加调度器
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=30, gamma=0.1)
        # 评价标准
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.fabric.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.fabric.device)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.fabric.device)

        # 为了记录到目前为止最好的验证准确性，方便保存最优模型
        self.val_acc_best = 0

        # tensorboard 记录器
        self.tb_log = TensorBoardLog
        self.tb_native_log = TensorBoardLog.experiment

        # tensorboard全局记录
        self.global_train_step = 0
        self.global_val_step = 0

        # # 默认记录net结构
        # 创建图像和文本输入张量
        image_input = torch.randn(tuple(net_input_size[0]))
        text_input = torch.randint(0, 100, tuple(net_input_size[1]))

        # 将图像和文本输入传递给模型
        self.tb_log.log_graph(self.net.cpu(), (image_input, text_input))  #
        summary(self.net.cpu(), input_data=(image_input, text_input), device="cpu")

        # 启动Fabric，覆盖当前变量
        self.fabric.launch()
        self.net, self.optim = self.fabric.setup(self.net, self.optim)

    def step(self, batch):
        img, text, y = batch
        if y.max() >= self.num_classes or y.min() < 0:
            raise ValueError(f"标签值超出范围: 最小值={y.min()}, 最大值={y.max()}, 类别数={self.num_classes}")
        text = text.to(self.fabric.device)
        img = img.to(self.fabric.device)
        y = y.to(self.fabric.device)
        pred = self.net(img, text).view(-1, self.num_classes)
        # targets = y.view(-1, 1)[:, 0]
        targets = y.view(-1).long()
        loss = self.loss(pred, targets)

        return loss, pred, targets

    def training(self, dataset_loader):
        dataset_loader = self.fabric.setup_dataloaders(dataset_loader)
        # 启动训练
        self.net.train()
        total_loss = 0

        # 进度条
        with tqdm(dataset_loader, desc="训练", colour="blue", leave=False) as t:
            for step, batch in enumerate(t):
                # 开始迭代
                self.optim.zero_grad()
                loss, pred, targets = self.step(batch)
                self.fabric.backward(loss)
                self.optim.step()
                # 计算指标
                self.train_acc(pred, targets)
                total_loss += loss.item()
                # 在每个步骤中更新进度条
                current_acc = self.train_acc.compute().item()
                t.set_postfix(loss=format(loss.item(), ".4f"), acc=format(current_acc, ".4f"))
                # 在每个步骤中记录到tensorborad中
                self.tb_native_log.add_scalar("train_loss", loss.item(), self.global_train_step)
                self.tb_native_log.add_scalar("train_acc", current_acc, self.global_train_step)
                self.global_train_step += 1

        avg_loss = total_loss / len(dataset_loader)
        acc = self.train_acc.compute()  # 使用自定义累积的所有批次的度量
        self.train_acc.reset()  # 重置内部状态，以便度量为新数据做好准备

        # 添加调度器步进，确保学习率在每个epoch结束后更新
        self.scheduler.step()
        return acc, avg_loss

    def validation(self, dataset_loader):
        dataset_loader = self.fabric.setup_dataloaders(dataset_loader)
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(dataset_loader, desc="验证", colour="green", leave=False) as t:
                for step, batch in enumerate(t):
                    loss, pred, targets = self.step(batch)
                    self.val_acc(pred, targets)
                    total_loss += loss.item()
                    # 在每个步骤中更新进度条
                    current_acc = self.val_acc.compute().item()
                    t.set_postfix(loss=format(loss.item(), ".4f"), acc=format(current_acc, ".4f"))
                    # 记录tb
                    self.tb_native_log.add_scalar("val_loss", loss.item(), self.global_val_step)
                    self.tb_native_log.add_scalar("val_acc", current_acc, self.global_val_step)

                    # 记录图像到 TensorBoard, 每个epoch只记录一次
                    if step == 0:
                        self.tb_native_log.add_images(
                            tag="I_images",
                            img_tensor=batch[0][:, :3, :, :],  # 假设 batch[0] 是图像数据
                            global_step=self.global_val_step,
                        )
                        self.tb_native_log.add_images(
                            tag="P_images",
                            img_tensor=batch[0][:, -3:, :, :],
                            global_step=self.global_val_step,
                        )
                    self.global_val_step += 1

        avg_loss = total_loss / len(dataset_loader)
        acc = self.val_acc.compute()  # 使用自定义累积的所有批次的度量
        self.val_acc.reset()  # 重置内部状态，以便度量为新数据做好准备
        # 对验证状态记录到tensorborad中

        return acc, avg_loss

    def save_model(self, save_path):
        checkpoint = {
            "net": self.net.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        log.info(f"Save Model, Path:{save_path}\n")
        self.fabric.save(save_path, checkpoint)

    def load_model(self, load_path, strict=True):
        """
        加载模型
        :param load_path: 加载路径
        :param strict: 加载检查点通常是“严格”的，这意味着检查点中的参数名称必须与模型中的参数名称匹配。 但是，在加载检查点进行微调或迁移学习时，可能会发生只有部分参数与模型匹配的情况。 对于这种情况，您可以禁用严格加载以避免错误：
        """
        if os.path.exists(load_path):
            try:
                full_checkpoint = self.fabric.load(load_path, strict=strict)
                log.info(f"=> Loaded pretrained model from {load_path} with loss: {full_checkpoint.get('loss', 'N/A')}")
                self.net.load_state_dict(full_checkpoint.get("net", {}), strict=strict)
                self.optim.load_state_dict(full_checkpoint.get("optimizer", {}))
            except Exception as e:
                log.error(f"Failed to load model from {load_path}: {e}")
        else:
            log.info(f"=> No checkpoint found at '{load_path}'")

    def init_weights(self):
        log.info("=> init Conv2d and BatchNorm2d ")
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def analytical_performance(self, dataset_loader):
        """
        用于测试模型性能
        :param dataset_loader: 需要测试的数据集
        """
        dataset_loader = self.fabric.setup_dataloaders(dataset_loader)
        # 评估模式
        self.net.eval()
        # tb性能分析器
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tb_log.log_dir),
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            # 只取一个数据作为测试对象
            with tqdm(dataset_loader, desc="性能测试", colour="yellow", leave=False) as t:
                for batch in t:
                    # 开始迭代
                    self.optim.zero_grad()
                    loss, pred, targets = self.step(batch)
                    self.fabric.backward(loss)
                    self.optim.step()
                    profiler.step()
        log.info("Profiler: \n{}".format(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)))  # profiler.key_averages().table(row_limit=10))
