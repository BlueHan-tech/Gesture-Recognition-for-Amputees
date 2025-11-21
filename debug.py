import scipy.io as sio
import numpy as np
import os
import glob

# 请修改为你的 Ninapro 数据集实际路径
ROOT_DATA_DIR = r"F:\A-SCI\Ninapro-DB3"


def check_raw_file():
    print(f"📂 正在搜索 E2 文件: {ROOT_DATA_DIR} ...")

    # 搜索文件名包含 E2 的 .mat 文件
    # 注意：DB3 的命名通常是 S1_E2_A1.mat
    pattern = os.path.join(ROOT_DATA_DIR, "**", "*E3*.mat")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("❌ 错误：在指定路径下没找到任何包含 'E3' 的 .mat 文件！")
        print("   请检查 ROOT_DATA_DIR 路径是否正确。")
        return

    # 选取第一个找到的文件进行检查
    target_file = files[5]
    print(f"🔎 正在检查文件: {target_file}")

    try:
        mat = sio.loadmat(target_file)

        # 尝试获取标签
        if 'restimulus' in mat:
            labels = mat['restimulus']
            label_name = 'restimulus'
        elif 'stimulus' in mat:
            labels = mat['stimulus']
            label_name = 'stimulus'
        else:
            print("❌ 文件中没有找到 stimulus 或 restimulus 标签字段。")
            print(f"   现有字段: {mat.keys()}")
            return

        unique_labels = np.unique(labels)
        count = len(unique_labels)
        if 0 in unique_labels:
            count -= 1  # 减去休息

        print(f"\n📊 原始文件分析报告:")
        print(f"   字段名: {label_name}")
        print(f"   包含标签: {unique_labels}")
        print(f"   非零类别数: {count}")

        print("\n⚖️  判决结果:")
        if count == 17:
            print("✅ 这是一个标准的 E2 文件 (17类)。")
            print("   -> 如果你之前跑出了23类，可能是脚本里混入了其他文件。")
        elif count == 23:
            print("🚨 这是一个 E3 文件 (23类)！！！")
            print("   -> 虽然文件名叫 E2，但内容是 E3。")
            print("   -> 建议重新下载数据集，或检查是否文件命名搞乱了。")
        else:
            print(f"⚠️ 类别数量 ({count}) 既不是 E2 也不是 E3，请核实数据来源。")

    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    check_raw_file()