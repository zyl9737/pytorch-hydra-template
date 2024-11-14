import os

import hydra
from lightning_fabric import fabric
from lightning_fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
from tqdm import trange

from scripts import debug_repr
from src.utils import get_logger, seed_torch

log = get_logger(__name__)
# os.environ["HYDRA_FULL_ERROR"] = "1"

debug_repr.enable_custom_repr()


@hydra.main(version_base=None, config_path="configs", config_name="experiment.yaml")
def main(config_global: DictConfig):
    """ """
    log.info(f"name: {config_global.name} version: {config_global.version} description: {config_global.description} ")
    # 初始化个性环境
    config = config_global.my_envs

    # 在pytorch，numpy和python中设置随机数生成器的种子
    seed_torch(config.train.get("seed"))
    fabric.seed_everything(config.train.get("seed"))
    log.info(f"seed is : <{config.train.get('seed')}>")

    # 必要时将相对ckpt路径转换为绝对路径
    config.train.resume_from_checkpoint = os.path.join(hydra.utils.get_original_cwd(), config.train.get("resume_from_checkpoint"))
    log.info(f"pretrained model path: <{config.train.resume_from_checkpoint}>")

    # 初始化数据加载器
    log.info(f"Initialize the DataModule: <{config.datamodule._target_}>\t path: <{config.datamodule.data_dir}>")
    train_dataloader = hydra.utils.instantiate(config.datamodule, mode="train").train_dataloader()
    # hydra.utils.instantiate(config.datamodule, mode="train")[0] # 测试数据__getter__函数是否正确
    val_dataloader = hydra.utils.instantiate(config.datamodule, mode="val").val_dataloader()

    # 初始化流程加载器
    log.info(f"Initialize the Trainer <{config.pipeline._target_}>")
    tb_logger = TensorBoardLogger(
        root_dir=f"/data/zyl/VMD/logs/runs/{config_global.name}/{config_global.version}/",
        name="TensorBoardLogger",
        flush_secs=10,
    )
    log.info(f"From the command line, use <\t tensorboard --logdir={os.path.abspath(tb_logger.log_dir)} \t> to view the current TensorBoard record.")
    train = hydra.utils.instantiate(config.pipeline, TensorBoardLog=tb_logger)

    # 测试训练性能
    if config_global.openPerformanceTest:
        log.info("start performance test!")
        train.analytical_performance(hydra.utils.instantiate(config.datamodule, mode="test_performance").train_dataloader())

    # 训练模型
    log.info("start training!")
    # train.load_model(load_path=config.train.resume_from_checkpoint)
    checkpoint_path = config.train.resume_from_checkpoint

    if os.path.exists(checkpoint_path):
        train.load_model(load_path=checkpoint_path)
    else:
        log.warning(f"No checkpoint found at '{checkpoint_path}'. Starting from scratch.")
        train.init_weights()  # 初始化模型权重
    epochs = config.train.epochs
    check_val_acc = 0
    # 使用 fabric.global_rank 只在主进程中显示进度条
    if train.fabric.global_rank == 0:
        t = trange(epochs, color="red")
    else:
        t = range(epochs)  # 非主进程不创建进度条

    for epoch in t:
        if train.fabric.global_rank == 0:
            t.set_description(f"总进度 Epoch {epoch}/{epochs}:")

        train_acc, train_loss = train.training(train_dataloader)
        val_acc, val_loss = train.validation(val_dataloader)

        if train.fabric.global_rank == 0:
            t.set_postfix(
                train_loss=format(train_loss, ".4f"),
                train_acc=format(train_acc, ".4f"),
                val_loss=format(val_loss, ".4f"),
                val_acc=format(val_acc, ".4f"),
            )

        # 记录到日志
        log.info(f"Epoch {epoch}/{epochs} : avg_train_loss={train_loss} avg_train_acc={train_acc} " f"avg_val_loss={val_loss}  avg_val_acc={val_acc}")

        # 记录到 TensorBoard
        tb_logger.log_metrics(
            {
                "avg_train_loss": train_loss,
                "avg_train_acc": train_acc,
                "avg_val_loss": val_loss,
                "avg_val_acc": val_acc,
            },
            step=epoch,
        )

        # 保存模型
        if train.fabric.global_rank == 0:
            # 确保保存目录存在
            resume_ckpt_path = config.train.get("resume_from_checkpoint")
            os.makedirs(os.path.dirname(resume_ckpt_path), exist_ok=True)

            # 保存最新模型到 'resume_from_checkpoint'
            train.save_model(save_path=resume_ckpt_path)
            log.info(f"保存最新模型到 {resume_ckpt_path}")

            # 如果验证准确率更高，保存到 'ckpt_path'
            if val_acc > check_val_acc:
                best_ckpt_path = config.train.get("ckpt_path")
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                train.save_model(save_path=best_ckpt_path)
                log.info(f"保存最佳验证准确率模型到 {best_ckpt_path}")
                check_val_acc = val_acc

    # 确保一切正常关闭
    log.info("Finalizing!")
    tb_logger.finalize("Finalizing")


if __name__ == "__main__":
    main()
