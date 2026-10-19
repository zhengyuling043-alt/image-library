"""
图像索引构建与 CLIP 模型集成模块

本模块实现了基于 CLIP（Contrastive Language-Image Pretraining）模型的图像索引构建和相似性搜索功能。
通过预先构建图像向量索引，可以大幅提升相似图像搜索的效率，避免每次搜索时重新编码所有素材图像。

核心功能：
- 使用 CLIP 模型（ViT-B-32-quickgelu 架构）将图像编码为高维向量
- 构建并持久化存储图像向量索引（NumPy 格式）
- 基于余弦相似度的快速图像检索

CLIP 模型说明：
- 模型架构：ViT-B-32-quickgelu（Vision Transformer Base，patch size 32，使用 quickgelu 激活函数）
- 预训练权重：openai（OpenAI 官方训练权重）
- 向量维度：512 维
- 特点：在保持较高精度的同时，推理速度优于 ViT-L-14 等更大模型

图像编码流程：
1. 加载图像并转换为 RGB 格式
2. 使用 CLIP 预处理器进行尺寸调整、裁剪和归一化
3. 通过模型编码为 512 维特征向量
4. 对向量进行 L2 归一化，使其模长为 1（便于后续余弦相似度计算）

使用方式：
- 构建索引：python indexed_search.py build
- 搜索图像：python indexed_search.py search --query path/to/query.jpg

注意：
- 当前实现使用 CPU 进行推理，如需使用 GPU 可修改 load_model() 中的 device 参数
- 索引文件存储在 data/index/ 目录下，包含向量文件和元数据 JSON
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

# 项目根目录
ROOT = Path(__file__).parent
# 素材图像存储目录
LIBRARY = ROOT / "data" / "library"
# 索引文件存储目录
INDEX_DIR = ROOT / "data" / "index"
# 将权重缓存在项目内的英文路径，避免旧版 PyTorch 在包含非 ASCII
# 用户名的 Windows 默认缓存目录中加载 TorchScript 权重失败。
MODEL_CACHE_DIR = ROOT / ".runtime" / "model-cache"
# CLIP 模型名称：ViT-B-32-quickgelu（Vision Transformer Base，patch size 32）
MODEL_NAME = "ViT-B-32-quickgelu"
# 预训练权重版本：openai 官方权重
WEIGHTS = "openai"
# 支持的图像文件扩展名
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def select_device():
    """Select CPU/GPU deterministically, with an environment override."""

    requested = os.environ.get("IMAGE_LIBRARY_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("IMAGE_LIBRARY_DEVICE 只能是 auto、cpu 或 cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已要求使用 CUDA，但当前 PyTorch 未检测到可用 GPU")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"


def load_model():
    """
    加载 CLIP 模型和图像预处理器
    
    功能：
        从 open_clip 库加载预训练的 CLIP 模型及其配套的图像预处理管道。
        模型加载后会自动设置为评估模式（eval mode），禁用 dropout 和 batch normalization
        的训练行为，确保推理结果的一致性。
    
    参数：
        无
    
    返回：
        tuple: (model, preprocess)
            - model: 加载好的 CLIP 模型实例，可用于图像编码
            - preprocess: 图像预处理函数，接受 PIL.Image 输出 torch.Tensor
    
    注意事项：
        - 当前硬编码使用 CPU 设备（device="cpu"）
        - 如需使用 GPU 加速，可将 device 改为 "cuda" 或 "mps"（Apple Silicon）
        - 模型首次加载时会从缓存或网络下载权重文件，可能需要一些时间
        - 返回的预处理函数包含：Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize
    
    示例：
        >>> model, preprocess = load_model()
        >>> # 模型已准备好进行图像编码
    """
    print("正在加载模型……")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from open_clip.pretrained import download_pretrained_from_url, get_pretrained_url

    weight_source = WEIGHTS
    weight_url = get_pretrained_url(MODEL_NAME, WEIGHTS)
    if weight_url:
        weight_source = download_pretrained_from_url(
            weight_url,
            cache_dir=str(MODEL_CACHE_DIR.resolve()),
        )
    # 使用 open_clip 创建模型和预处理管道
    # create_model_and_transforms 返回三元组：(模型, tokenizer, 预处理器)
    # 这里我们只需要模型和预处理器，因此用 _ 忽略 tokenizer
    device = select_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=weight_source,
        device=device,
        cache_dir=str(MODEL_CACHE_DIR.resolve()),
        # OpenAI 官方 .pt 是 TorchScript 归档；PyTorch 2.6+ 的默认
        # weights_only=True 无法读取。weight_source 仅来自官方预训练URL缓存。
        weights_only=False,
    )
    # 将模型设置为评估模式，关闭训练专用层（如 dropout）
    model.eval()
    return model, preprocess


def encode_image(path, model, preprocess):
    """
    将单张图像编码为 CLIP 特征向量
    
    功能：
        读取指定路径的图像文件，通过 CLIP 模型将其编码为 512 维的归一化特征向量。
        该向量可用于后续的相似度计算和图像检索。
    
    参数：
        path (Path): 图像文件的路径，支持 JPG、PNG、WEBP、BMP 等常见格式
        model: 已加载的 CLIP 模型实例
        preprocess: 图像预处理函数，由 load_model() 返回
    
    返回：
        numpy.ndarray: 形状为 (512,) 的一维数组，表示图像的 CLIP 特征向量
            - 向量已经过 L2 归一化，模长为 1
            - 数据类型为 float32
            - 可直接用于余弦相似度计算（向量点积）
    
    异常处理：
        OSError: 图像文件无法读取或损坏时抛出
        ValueError: 图像格式不支持或预处理失败时抛出
    
    注意事项：
        - 图像会自动转换为 RGB 模式，灰度图或 RGBA 图会被正确处理
        - 使用 torch.inference_mode() 上下文管理器禁用梯度计算，节省内存并加速推理
        - 向量归一化使用 L2 范数，确保所有向量位于单位超球面上
        - 最终结果会先转移到 CPU 再转换为 NumPy 数组，兼容无 GPU 环境
    
    示例：
        >>> model, preprocess = load_model()
        >>> vector = encode_image(Path("image.jpg"), model, preprocess)
        >>> vector.shape
        (512,)
    """
    # 打开图像文件并转换为 RGB 格式（处理灰度图、RGBA 等情况）
    with Image.open(path) as image:
        # preprocess 执行以下操作：
        # 1. Resize: 将短边缩放到 224 像素
        # 2. CenterCrop: 中心裁剪 224x224 区域
        # 3. ToTensor: 转换为 [0, 1] 范围的张量
        # 4. Normalize: 使用 CLIP 专用均值和标准差进行归一化
        # unsqueeze(0) 添加批次维度，形状从 (3, 224, 224) 变为 (1, 3, 224, 224)
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    # 使用推理模式上下文管理器，禁用梯度计算和某些训练专用操作
    # 相比 torch.no_grad()，inference_mode() 性能更优
    with torch.inference_mode():
        # 通过 CLIP 模型的图像编码器将图像张量编码为特征向量
        # 输出形状为 (1, 512)，512 是 ViT-B-32 的嵌入维度
        vector = model.encode_image(tensor)
        # L2 归一化：将向量除以其模长，使结果向量模长为 1
        # 归一化后的向量之间的点积等于余弦相似度
        # dim=-1 表示对最后一个维度（特征维度）计算范数
        # keepdim=True 保持维度以便广播除法
        vector = vector / vector.norm(dim=-1, keepdim=True)

    # 将结果转移到 CPU 并转换为 NumPy 数组
    # [0] 索引移除批次维度，最终形状为 (512,)
    return vector[0].cpu().numpy()


def build_index():
    """
    构建素材库的完整向量索引
    
    功能：
        扫描素材库目录中的所有图像文件，使用 CLIP 模型将它们编码为向量，
        并将结果保存到磁盘。构建好的索引可以被 search() 函数快速加载，
        避免每次搜索时重新编码所有素材。
    
    参数：
        无
    
    返回：
        无
    
    异常：
        RuntimeError: 以下情况会抛出异常：
            - 素材库目录为空或不存在支持的图像文件
            - 所有图像都无法成功处理（损坏、格式错误等）
    
    输出文件：
        1. vectors.npy: NumPy 二进制格式，形状为 (N, 512) 的浮点数组
           - N 是成功处理的图像数量
           - 每一行对应一张图像的 CLIP 特征向量
        2. metadata.json: JSON 格式的元数据文件，包含：
           - model: 使用的模型名称
           - weights: 预训练权重版本
           - files: 图像文件名列表（与向量矩阵的行一一对应）
    
    注意事项：
        - 索引构建时间与素材数量成正比，首次构建可能需要几分钟
        - 处理过程中会打印进度信息
        - 损坏或无法读取的图像会被跳过，不会中断整个流程
        - 如果素材库更新，需要重新执行此函数以更新索引
        - 索引文件存储位置由 INDEX_DIR 常量定义
    
    示例：
        >>> build_index()
        正在加载模型……
        处理 1/100：image001.jpg
        处理 2/100：image002.jpg
        ...
        索引建立成功！已保存 100 张图片的向量。
    """
    # 扫描素材库目录，筛选出支持格式的图像文件
    # sorted() 确保处理顺序一致，便于调试和进度跟踪
    files = sorted(
        path for path in LIBRARY.glob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    # 检查是否找到任何图像文件
    if not files:
        raise RuntimeError(f"素材库没有图片：{LIBRARY}")

    # 加载 CLIP 模型和预处理器
    model, preprocess = load_model()
    # names: 存储成功处理的图像文件名
    # vectors: 存储对应的特征向量（列表形式，后续转换为数组）
    names, vectors = [], []

    # 遍历所有图像文件，逐个编码为向量
    # enumerate(start=1) 用于显示从 1 开始的序号
    for number, path in enumerate(files, start=1):
        print(f"处理 {number}/{len(files)}：{path.name}")
        try:
            # 调用 encode_image 将图像编码为 512 维向量
            vector = encode_image(path, model, preprocess)
        except (OSError, ValueError) as error:
            # 捕获图像读取或处理错误，跳过该文件继续处理
            # OSError: 文件损坏、权限问题等
            # ValueError: 格式不支持、尺寸异常等
            print(f"跳过无法读取的图片：{path.name}，原因：{error}")
            continue

        # 将成功处理的文件名和向量添加到列表
        names.append(path.name)
        vectors.append(vector)

    # 检查是否有成功处理的图像
    if not vectors:
        raise RuntimeError("没有成功处理的图片，未保存索引。")

    # 创建索引目录（如果不存在）
    # parents=True 允许创建多级目录，exist_ok=True 避免目录已存在时报错
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # 将向量列表堆叠为二维数组并保存为 NumPy 二进制文件
    # np.stack(vectors) 将形状为 (512,) 的 N 个向量堆叠为 (N, 512) 的数组
    np.save(INDEX_DIR / "vectors.npy", np.stack(vectors))

    # 构建元数据字典，记录索引的相关信息
    metadata = {
        "model": MODEL_NAME,      # 模型名称，用于加载时验证
        "weights": WEIGHTS,       # 权重版本，确保一致性
        "files": names,           # 文件名列表，与向量矩阵的行对应
    }
    # 将元数据保存为 JSON 文件
    # ensure_ascii=False 允许存储中文等非 ASCII 字符
    # indent=2 使输出格式化，便于阅读
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n索引建立成功！已保存 {len(names)} 张图片的向量。")


def search(query_path):
    """
    基于向量索引的相似图像搜索
    
    功能：
        加载预先构建的向量索引，将查询图像编码为向量，
        通过余弦相似度计算找到最相似的 Top-5 素材图像。
        由于使用了预构建索引，搜索过程只需编码查询图像一次，
        大幅提升了检索效率。
    
    参数：
        query_path (Path): 查询图像的路径，支持常见图像格式
    
    返回：
        无（结果直接打印到控制台）
    
    异常：
        RuntimeError: 以下情况会抛出异常：
            - 索引文件不存在（需要先执行 build_index）
            - 索引使用的模型与当前配置不匹配
            - 索引文件损坏或数据不一致
        FileNotFoundError: 查询图像文件不存在
    
    工作原理：
        1. 加载索引：从磁盘读取向量矩阵和元数据
        2. 验证一致性：检查模型名称和权重版本是否匹配
        3. 编码查询：使用 CLIP 模型将查询图像编码为向量
        4. 计算相似度：通过矩阵乘法计算查询向量与所有素材向量的点积
           （由于向量已归一化，点积等于余弦相似度）
        5. 排序输出：按相似度降序排列，输出 Top-5 结果
    
    注意事项：
        - 搜索前必须先执行 build_index() 构建索引
        - 如果模型或权重发生变化，需要重新构建索引
        - 相似度范围：[-1, 1]，值越接近 1 表示越相似
        - 当前实现返回 Top-5 结果，可根据需要修改数量
    
    示例：
        >>> search(Path("query.jpg"))
        已读取 100 张素材的向量，无需重新编码素材。
        正在加载模型……
        
        查询图片：query.jpg
        最相似的前 5 张图片：
        1. 相似度 0.9523 | similar1.jpg
        2. 相似度 0.9187 | similar2.jpg
        ...
    """
    # 定义索引文件路径
    vector_file = INDEX_DIR / "vectors.npy"
    metadata_file = INDEX_DIR / "metadata.json"

    # 检查索引文件是否存在
    if not vector_file.is_file() or not metadata_file.is_file():
        raise RuntimeError("请先执行：python indexed_search.py build")

    # 检查查询图像是否存在
    if not query_path.is_file():
        raise FileNotFoundError(f"找不到查询图片：{query_path}")

    # 加载元数据并验证模型一致性
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    # 确保索引使用的模型和权重与当前配置一致
    # 如果不一致，向量空间不同，搜索结果将无效
    if (
        metadata["model"] != MODEL_NAME
        or metadata["weights"] != WEIGHTS
    ):
        raise RuntimeError("索引使用的模型已变化，请重新建立索引。")

    # 加载向量索引文件
    # allow_pickle=False 禁用 pickle 反序列化，提高安全性
    vectors = np.load(vector_file, allow_pickle=False)
    # 从元数据中获取文件名列表
    names = metadata["files"]
    # 验证向量矩阵和文件名列表的一致性
    if vectors.ndim != 2 or len(vectors) != len(names):
        raise RuntimeError("索引文件不一致，请重新建立索引。")

    print(f"已读取 {len(names)} 张素材的向量，无需重新编码素材。")
    # 加载 CLIP 模型用于编码查询图像
    model, preprocess = load_model()
    # 将查询图像编码为归一化的 512 维向量
    query_vector = encode_image(query_path, model, preprocess)
    
    # 计算余弦相似度（由于向量已归一化，点积即为余弦相似度）
    # vectors 形状为 (N, 512)，query_vector 形状为 (512,)
    # 矩阵乘法结果 scores 形状为 (N,)，每个元素是对应素材的相似度分数
    scores = vectors @ query_vector
    # 获取相似度最高的前 5 个索引
    # np.argsort(-scores) 按相似度降序排列返回索引
    # [:5] 取前 5 个（如果素材不足 5 张则返回全部）
    order = np.argsort(-scores)[:5]

    # 输出搜索结果
    print(f"\n查询图片：{query_path.name}")
    print("最相似的前 5 张图片：")
    # 按排名顺序输出结果
    for rank, index in enumerate(order, start=1):
        print(f"{rank}. 相似度 {scores[index]:.4f} | {names[index]}")


# 命令行入口点：当直接运行此脚本时执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加位置参数 action，只能是 "build" 或 "search"
    # build: 构建素材库的向量索引
    # search: 执行相似图像搜索
    parser.add_argument("action", choices=["build", "search"])
    # 添加可选参数 --query，指定查询图像的路径
    # 默认值为 "data/test.jpg"，仅在 search 操作时使用
    parser.add_argument("--query", default="data/test.jpg")
    # 解析命令行参数
    args = parser.parse_args()

    # 根据用户选择的操作执行相应功能
    if args.action == "build":
        # 构建索引：扫描素材库并生成向量文件
        build_index()
    else:
        # 搜索操作：处理查询图像路径
        query_path = Path(args.query)
        # 如果提供的是相对路径，转换为相对于项目根目录的绝对路径
        if not query_path.is_absolute():
            query_path = ROOT / query_path
        # 执行相似图像搜索
        search(query_path)
