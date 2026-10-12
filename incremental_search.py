"""
增量图片搜索索引模块。

本模块实现了基于文件哈希的增量索引机制，用于高效管理图片向量数据库。

核心功能：
    1. 增量索引更新：通过 SHA256 哈希检测文件变化，仅重新编码修改过的图片
    2. 向量编码：使用 CLIP 模型将图片转换为 512 维特征向量
    3. 原子保存：使用临时文件 + 原子替换确保索引文件的完整性
    4. 相似度搜索：基于余弦相似度快速检索最相似的图片

工作原理：
    - 文件哈希检测：对每个图片文件计算 SHA256 哈希值，与索引中记录的哈希对比
    - 向量编码：使用 OpenAI CLIP 模型 (ViT-B-32) 提取图片特征
    - 增量更新：只重新编码哈希值变化的图片，未变化的直接复用旧向量
    - 持久化存储：将向量和元数据保存为 NumPy .npz 格式

使用示例：
    # 命令行使用
    python incremental_search.py update              # 更新索引
    python incremental_search.py update --rebuild    # 重建索引
    python incremental_search.py search --query data/test.jpg  # 搜索相似图片
    
    # 代码调用
    from incremental_search import update_index, search
    from pathlib import Path
    
    # 更新索引
    report = update_index()
    print(f"已索引 {report['ready']} 张图片")
    
    # 搜索相似图片
    query_path = Path("data/test.jpg")
    search(query_path)

注意事项：
    - 首次使用需要运行 update 命令建立索引
    - 索引文件保存在 data/index/incremental.npz
    - 模型配置变更时需要使用 --rebuild 重建索引
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

# ==================== 全局常量配置 ====================
# 项目根目录，用于构建相对路径
ROOT = Path(__file__).parent
# 图片库目录，存放待索引的图片文件
LIBRARY = ROOT / "data" / "library"
# 索引文件路径，保存向量数据和元数据
INDEX_FILE = ROOT / "data" / "index" / "incremental.npz"
# CLIP 模型名称，ViT-B-32 是轻量级视觉Transformer模型
MODEL = "ViT-B-32-quickgelu"
# 模型权重来源，openai 表示使用 OpenAI 官方预训练权重
WEIGHTS = "openai"
# 支持的图片格式扩展名集合
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fingerprint(path):
    """
    计算文件的 SHA256 哈希值，用于检测文件是否发生变化。
    
    参数:
        path (Path): 要计算哈希的文件路径对象
        
    返回:
        str: 文件的 SHA256 哈希值（十六进制字符串，长度64）
        
    工作原理:
        - 使用 SHA256 算法生成文件的唯一指纹
        - 分块读取（每次 1MB）避免大文件占用过多内存
        - 使用 iter() 的哨兵值模式实现循环读取直到文件末尾
        
    示例:
        >>> from pathlib import Path
        >>> hash_value = fingerprint(Path("image.jpg"))
        >>> print(len(hash_value))  # 输出: 64
    """
    # 初始化 SHA256 哈希对象
    digest = hashlib.sha256()
    # 以二进制只读模式打开文件
    with path.open("rb") as stream:
        # 分块读取文件，每次读取 1MB，避免大文件内存溢出
        # iter(callable, sentinel) 模式：不断调用 callable 直到返回 sentinel 值
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            # 更新哈希值，支持增量计算
            digest.update(block)
    # 返回十六进制格式的哈希字符串
    return digest.hexdigest()


class Encoder:
    """
    图片向量编码器，使用懒加载模式延迟加载 CLIP 模型。
    
    该类封装了图片到向量的编码过程，核心特性是懒加载机制：
    只有在真正需要编码图片时才加载模型，避免不必要的内存占用和
    模型下载操作。这在增量更新场景中特别有用——如果没有文件变化，
    模型根本不会被加载。
    
    属性:
        model: CLIP 模型实例，初始为 None，首次调用时加载
        preprocess: 图片预处理函数，与模型一起加载
        
    使用示例:
        >>> encoder = Encoder()
        >>> # 此时模型尚未加载，不占用内存
        >>> vector = encoder(Path("image.jpg"))
        >>> # 首次调用时自动加载模型
        >>> print(vector.shape)  # 输出: (512,)
        
    注意事项:
        - 模型加载后会缓存在实例中，后续调用不会重复加载
        - 每个 Encoder 实例独立维护自己的模型缓存
        - 模型来自 indexed_search 模块的 load_model 函数
    """
    
    def __init__(self):
        """
        初始化编码器，此时不加载模型。
        
        将模型和预处理函数设置为 None，延迟到首次编码时再加载。
        这种设计避免了在不需要编码的场景（如纯搜索）中浪费资源。
        """
        self.model = None
        self.preprocess = None  # 图片预处理函数，与模型配套

    def __call__(self, path):
        """
        将图片文件编码为 512 维特征向量。
        
        参数:
            path (Path): 图片文件路径
            
        返回:
            numpy.ndarray: 512 维浮点向量，表示图片的特征嵌入
            
        副作用:
            - 首次调用时会加载 CLIP 模型到内存
            - 会从 indexed_search 模块导入 load_model 和 encode_image
            
        工作原理:
            1. 检查模型是否已加载（懒加载检查）
            2. 如果未加载，调用 load_model() 加载模型和预处理函数
            3. 调用 encode_image() 将图片转换为向量
            
        注意事项:
            - 导入语句放在函数内部，避免模块加载时就依赖 indexed_search
            - 模型加载后缓存在 self.model，后续调用直接使用
        """
        # 延迟导入：避免模块加载时就触发模型下载平台访问
        from indexed_search import load_model, encode_image
        
        # 懒加载检查：首次调用时才加载模型
        if self.model is None:
            self.model, self.preprocess = load_model()
        
        # 使用加载的模型编码图片，返回特征向量
        return encode_image(path, self.model, self.preprocess)


def read_index(path):
    """
    从 .npz 文件读取索引数据，包括元数据和向量。
    
    参数:
        path (Path): 索引文件路径（.npz 格式）
        
    返回:
        tuple: (records, vectors) 元组
            - records (dict): 元数据字典，格式为 {文件名: {sha256: 哈希值, status: 状态}}
            - vectors (dict): 向量字典，格式为 {文件名: numpy向量}
            
    异常:
        RuntimeError: 当出现以下情况时抛出
            - 模型配置与当前不匹配（需要重建索引）
            - 向量维度与记录数量不一致
            - 向量包含非有限值（NaN 或 Inf）
            
    工作原理:
        1. 检查索引文件是否存在，不存在返回空字典
        2. 使用 numpy.load 加载 .npz 文件
        3. 从 metadata 字段解析 JSON 格式的元数据
        4. 从 vectors 字段读取向量数组
        5. 验证模型配置、维度一致性和向量有效性
        6. 将向量数组转换为 {文件名: 向量} 字典返回
        
    注意事项:
        - allow_pickle=False 防止反序列化攻击
        - 元数据以 JSON 字符串形式存储在 numpy 数组中
        - 向量必须全部是有限值，否则说明编码过程出错
        
    示例:
        >>> records, vectors = read_index(Path("index.npz"))
        >>> print(f"已加载 {len(records)} 条记录")
    """
    # 索引文件不存在时返回空结果，表示首次构建
    if not path.exists():
        return {}, {}
    
    # 加载 .npz 文件，allow_pickle=False 防止恶意 pickle 攻击
    with np.load(path, allow_pickle=False) as archive:
        # metadata 字段存储 JSON 字符串，需要先转为 Python 字符串再解析
        metadata = json.loads(str(archive["metadata"].item()))
        # vectors 字段存储所有图片的特征向量（二维数组）
        vectors = archive["vectors"]
    
    # 验证模型配置：确保索引是用当前模型配置构建的
    if (metadata.get("model"), metadata.get("weights")) != (MODEL, WEIGHTS):
        raise RuntimeError("模型配置不同，请使用 --rebuild 重建增量索引。")
    
    # 提取记录字典：{文件名: {sha256: 哈希, status: 状态}}
    records = metadata["records"]
    names = list(records)
    
    # 验证向量维度：必须是二维数组，且行数等于记录数
    if vectors.ndim != 2 or len(names) != len(vectors):
        raise RuntimeError("增量索引不一致，请使用 --rebuild 重建。")
    
    # 验证向量有效性：所有值必须是有限值（非 NaN、非 Inf）
    if not np.isfinite(vectors).all():
        raise RuntimeError("索引含无效向量，请重新建立。")
    
    # 将向量数组转换为字典：{文件名: 对应的向量}
    # zip 将文件名列表和向量数组按行配对
    return records, dict(zip(names, vectors))


def save_index(path, records, vectors):
    """
    原子性地保存索引到 .npz 文件，确保数据完整性。
    
    参数:
        path (Path): 索引文件保存路径
        records (dict): 元数据字典，格式为 {文件名: {sha256: 哈希值, status: 状态}}
        vectors (dict): 向量字典，格式为 {文件名: numpy向量}
        
    副作用:
        - 创建父目录（如果不存在）
        - 写入索引文件（.npz 格式）
        - 使用临时文件 + 原子替换，确保写入失败时不影响旧索引
        
    工作原理:
        1. 将向量字典堆叠为二维 numpy 数组
        2. 将元数据序列化为 JSON 字符串
        3. 创建临时文件，写入向量和元数据
        4. 强制刷新到磁盘（fsync）
        5. 使用 os.replace 原子替换目标文件
        
    原子性保证:
        - 使用 tempfile.NamedTemporaryFile 在同目录创建临时文件
        - 写入完成后调用 fsync 确保数据落盘
        - 使用 os.replace 原子替换，旧文件在替换前始终可用
        - finally 块确保临时文件被清理
        
    注意事项:
        - 空索引保存为 (0, 512) 形状的空数组，保持维度一致
        - ensure_ascii=False 允许 JSON 中包含非 ASCII 字符
        - 临时文件使用 .npz 后缀，与目标文件相同
        
    示例:
        >>> records = {"img.jpg": {"sha256": "abc123", "status": "ready"}}
        >>> vectors = {"img.jpg": np.random.randn(512)}
        >>> save_index(Path("index.npz"), records, vectors)
    """
    # 确保父目录存在，parents=True 递归创建，exist_ok=True 忽略已存在错误
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将向量字典堆叠为二维数组：(N, 512)
    # 如果 records 为空，创建形状为 (0, 512) 的空数组
    array = np.stack([vectors[name] for name in records]) if records else np.empty((0, 512), dtype=np.float32)
    
    # 将元数据序列化为 JSON 字符串，包含模型配置和记录信息
    metadata = json.dumps({"model": MODEL, "weights": WEIGHTS, "records": records}, ensure_ascii=False)
    
    # 原子保存：使用临时文件 + 原子替换，确保写入失败不影响旧索引
    temporary = None
    try:
        # 在同目录创建临时文件，delete=False 防止自动删除（我们需要手动控制）
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            # 保存向量和元数据到 .npz 文件
            # metadata 需要包装为 numpy 数组才能保存
            np.savez(stream, vectors=array, metadata=np.array(metadata))
            # 刷新用户态缓冲区到内核
            stream.flush()
            # 强制刷新到磁盘，确保断电时数据不丢失
            os.fsync(stream.fileno())
        
        # 原子替换：在 Windows 上也是原子操作，旧文件在替换前始终可用
        os.replace(temporary, path)
    finally:
        # 确保临时文件被清理，missing_ok=True 忽略文件不存在的错误
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_index(library=LIBRARY, index_file=INDEX_FILE, encoder=None, rebuild=False):
    """
    增量更新图片索引，只重新编码发生变化的图片。
    
    这是本模块的核心函数，实现了高效的增量索引机制：
    1. 通过 SHA256 哈希检测文件是否变化
    2. 未变化的文件直接复用旧向量（节省计算时间）
    3. 新增或修改的文件重新编码
    4. 删除的文件自动从索引中移除
    
    参数:
        library (Path): 图片库目录路径，默认为 data/library
        index_file (Path): 索引文件路径，默认为 data/index/incremental.npz
        encoder (Encoder, optional): 编码器实例，为 None 时自动创建
        rebuild (bool): 是否强制重建索引（忽略旧缓存），默认 False
        
    返回:
        dict: 更新报告，包含以下字段：
            - counts (dict): 统计信息
                - added (int): 新增图片数量
                - modified (int): 修改图片数量
                - reused (int): 复用向量数量
                - deleted (int): 删除图片数量
                - failed (int): 处理失败数量
            - ready (int): 成功索引的图片总数
            - failures (dict): 失败记录，格式为 {文件名: 错误信息}
            
    异常:
        RuntimeError: 当图片库目录不存在时抛出
        
    副作用:
        - 读取并更新索引文件
        - 生成 .report.json 报告文件
        - 控制台输出更新进度和统计信息
        
    增量逻辑详解:
        1. 读取旧索引（如果 rebuild=False）
        2. 遍历图片库中的所有图片文件
        3. 对每个文件计算 SHA256 哈希
        4. 与旧索引中的哈希对比：
           - 哈希相同 → 复用旧向量（最快）
           - 哈希不同 → 重新编码（新增或修改）
        5. 旧索引中有但当前不存在的文件自动删除
        6. 原子保存新索引
        
    示例:
        >>> report = update_index()
        >>> print(f"新增 {report['counts']['added']} 张")
        >>> print(f"总共 {report['ready']} 张可搜索")
    """
    # 验证图片库目录存在
    if not library.is_dir():
        raise RuntimeError(f"素材文件夹不存在：{library}，请检查路径。")
    
    # 读取旧索引：rebuild=True 时清空旧数据，否则从文件加载
    old_records, old_vectors = ({}, {}) if rebuild else read_index(index_file)
    
    # 扫描图片库：获取所有支持的图片文件，按文件名排序确保顺序稳定
    files = sorted(p for p in library.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    
    # 当前图片库中的文件名集合，用于计算删除数量
    current_names = {p.name for p in files}
    
    # 初始化统计计数器
    # deleted 计算：旧索引中有但当前不存在的文件数量
    counts = dict(added=0, modified=0, reused=0, deleted=len(set(old_records) - current_names), failed=0)
    
    # 新的索引数据：records 存储元数据，vectors 存储向量，failures 存储失败记录
    records, vectors, failures = {}, {}, {}
    
    # 创建或复用编码器：允许外部传入自定义编码器（主要用于测试）
    encoder = encoder or Encoder()
    
    # 遍历所有图片文件，执行增量更新
    for path in files:
        name = path.name
        try:
            # 第一步：计算当前文件的 SHA256 哈希
            digest = fingerprint(path)
            
            # 第二步：检查是否与旧索引中的哈希匹配
            # old_records.get(name, {}).get("sha256") 安全地获取旧哈希值
            if old_records.get(name, {}).get("sha256") == digest:
                # 哈希匹配：文件未变化，直接复用旧向量
                vector = old_vectors[name]
                counts["reused"] += 1
            else:
                # 哈希不匹配：文件是新增或已修改，需要重新编码
                print(f"计算图片向量：{name}")
                
                # 调用编码器将图片转换为 512 维向量
                vector = np.asarray(encoder(path), dtype=np.float32)
                
                # 验证向量有效性：必须是 512 维且所有值有限
                if vector.shape != (512,) or not np.isfinite(vector).all():
                    raise ValueError("模型输出不是有效的 512 维向量")
                
                # 一致性检查：编码后再次计算哈希，确保文件在处理期间未被修改
                # 这是一个竞态条件检测，防止读取到不一致的数据
                if fingerprint(path) != digest:
                    raise ValueError("图片在处理期间发生变化，请稍后重试")
                
                # 更新统计：根据文件名是否在旧索引中，区分"新增"和"修改"
                counts["modified" if name in old_records else "added"] += 1
            
            # 第三步：保存成功处理的文件信息
            records[name] = {"sha256": digest, "status": "ready"}
            vectors[name] = vector
            
        except (OSError, ValueError) as error:
            # 处理失败：记录错误信息，继续处理下一个文件
            counts["failed"] += 1
            failures[name] = str(error)
            print(f"入库失败：{name}：{error}")
            # 失败的文件不会加入新索引，下次 update 时会自动重试
    
    # 第四步：原子保存新索引
    save_index(index_file, records, vectors)
    
    # 第五步：生成并保存报告文件
    report = {"counts": counts, "ready": len(records), "failures": failures}
    index_file.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    
    # 第六步：输出统计信息到控制台
    print(f"\n新增 {counts['added']}，修改 {counts['modified']}，复用 {counts['reused']}，删除 {counts['deleted']}，失败 {counts['failed']}")
    print(f"当前可搜索图片：{len(records)} 张")
    
    return report


def search(query, library=LIBRARY, index_file=INDEX_FILE):
    """
    在索引中搜索与查询图片最相似的图片。
    
    使用余弦相似度（对于归一化向量等价于点积）计算查询图片与
    索引中所有图片的相似度，返回最相似的前5张。
    
    参数:
        query (Path): 查询图片的路径
        library (Path): 图片库目录路径，默认为 data/library
        index_file (Path): 索引文件路径，默认为 data/index/incremental.npz
        
    返回:
        None: 结果直接打印到控制台，不返回值
        
    异常:
        RuntimeError: 当索引文件不存在时抛出（需要先运行 update）
        FileNotFoundError: 当查询图片不存在时抛出
        
    工作原理:
        1. 加载索引中的所有向量
        2. 使用 Encoder 编码查询图片为向量
        3. 对索引中的每个向量，计算与查询向量的点积（相似度）
        4. 按相似度降序排序，取前5名
        5. 打印结果到控制台
        
    过期检测:
        - 搜索时会检查索引中的文件是否仍然有效
        - 如果文件的 SHA256 哈希与索引记录不符，标记为"过期"
        - 过期的文件不参与搜索，并提示用户运行 update 同步
        
    注意事项:
        - 搜索前必须先运行 update 建立索引
        - 只编码查询图片，不重新编码索引中的图片（高效）
        - 相似度分数范围：[-1, 1]，越接近1表示越相似
        
    示例:
        >>> from pathlib import Path
        >>> search(Path("data/test.jpg"))
        读取已保存的 100 张素材向量，只编码查询图。
        
        查询图片：test.jpg；最相似的前 5 张：
        1. 相似度 0.9523 | image1.jpg
        2. 相似度 0.8912 | image2.jpg
        ...
    """
    # 检查索引文件是否存在
    if not index_file.is_file():
        raise RuntimeError("请先运行 python incremental_search.py update")
    
    # 检查查询图片是否存在
    if not query.is_file():
        raise FileNotFoundError(f"查询图不存在：{query}")
    
    # 加载索引数据
    records, vectors = read_index(index_file)
    
    # 检查索引是否为空
    if not records:
        print("索引为空，请加入图片后运行 update。")
        return
    
    print(f"读取已保存的 {len(records)} 张素材向量，只编码查询图。")
    
    # 编码查询图片：创建临时 Encoder 实例并调用
    query_vector = Encoder()(query)
    
    # 存储搜索结果：(相似度分数, 文件名) 元组列表
    results = []
    # 过期文件计数器：哈希不匹配或文件不存在的记录
    stale = 0
    
    # 遍历索引中的所有记录，计算相似度
    for name, record in records.items():
        try:
            # 验证文件是否过期：重新计算哈希并与索引记录对比
            if fingerprint(library / name) != record["sha256"]:
                stale += 1
                continue  # 跳过过期文件
        except OSError:
            # 文件不存在或无法读取，标记为过期
            stale += 1
            continue
        
        # 计算余弦相似度（对于归一化向量，点积等价于余弦相似度）
        # @ 运算符执行向量点积：query_vector · vectors[name]
        results.append((float(vectors[name] @ query_vector), name))
    
    # 如果有过期文件，提示用户更新索引
    if stale:
        print(f"已跳过 {stale} 张已删除或变化的素材，请运行 update 同步。")
    
    # 打印搜索结果
    print(f"\n查询图片：{query.name}；最相似的前 5 张：")
    
    # 排序规则：先按相似度降序（-x[0]），再按文件名升序（x[1]）
    # 取前5名，从1开始编号
    for rank, (score, name) in enumerate(sorted(results, key=lambda x: (-x[0], x[1]))[:5], 1):
        print(f"{rank}. 相似度 {score:.4f} | {name}")


if __name__ == "__main__":
    # ==================== 命令行入口 ====================
    # 创建参数解析器，使用模块文档字符串作为描述
    parser = argparse.ArgumentParser(description=__doc__)
    
    # 位置参数：操作类型，只能是 update 或 search
    parser.add_argument("action", choices=["update", "search"])
    
    # 可选参数：查询图片路径，默认为 data/test.jpg
    parser.add_argument("--query", default="data/test.jpg")
    
    # 可选参数：是否重建索引（忽略旧缓存）
    parser.add_argument("--rebuild", action="store_true", help="放弃旧增量缓存并重新编码")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据操作类型执行对应函数
    if args.action == "update":
        # 更新索引：rebuild=True 时强制重建，否则增量更新
        update_index(rebuild=args.rebuild)
    else:
        # 搜索模式：处理查询图片路径
        query = Path(args.query)
        # 如果是相对路径，转换为相对于项目根目录的绝对路径
        search(query if query.is_absolute() else ROOT / query)
