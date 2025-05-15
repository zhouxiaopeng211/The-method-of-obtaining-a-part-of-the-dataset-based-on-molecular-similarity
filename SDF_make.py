import SDF_dispose
from rdkit import Chem
from tqdm import tqdm

def is_valid_molecule(mol):
    """检查分子是否有效"""
    if mol is None:
        return False
    graph = SDF_dispose.molecule_to_pyg_graph(mol)
    if len(graph.edge_index) == 0:
        return False
    try:
        # 检查分子是否可净化（化学合理性）
        Chem.SanitizeMol(mol)
        # 确保分子至少包含一个原子
        return mol.GetNumAtoms() > 0
    except:
        return False


def filter_sdf(input_file, output_file):
    """过滤无效分子并保存到新文件"""
    # 读取SDF文件
    suppl = Chem.SDMolSupplier(input_file)
    # 创建输出文件写入器
    writer = Chem.SDWriter(output_file)

    skipped_count = 0
    total_count = 0

    for idx, mol in enumerate(suppl):
        total_count += 1
        try:
            if is_valid_molecule(mol) :
                writer.write(mol)
            else:
                skipped_count += 1
                print(f"跳过无效分子 #{idx + 1}")

        except Exception as e:
            skipped_count += 1
            print(f"处理分子 #{idx + 1} 时发生错误: {str(e)}")

    writer.close()
    print(f"\n处理完成！共处理 {total_count} 个分子")
    print(f"保留有效分子: {total_count - skipped_count}")
    print(f"删除无效分子: {skipped_count}")

def remove_nth_molecule(input_file, output_file, n):
    """删除指定索引的分子（索引从 0 开始）"""
    supplier = Chem.SDMolSupplier(input_file)
    writer = Chem.SDWriter(output_file)

    for idx, mol in enumerate(supplier):
        if idx != n:  # 跳过第n个分子
            if mol:  # 同时检查有效性
                writer.write(mol)

    writer.close()



def add_name_property_to_sdf(input_sdf, output_sdf):
    """
    为SDF文件中的每个分子添加name属性，按顺序命名为name0, name1, ...

    参数:
    input_sdf -- 输入SDF文件路径
    output_sdf -- 输出SDF文件路径
    """
    # 读取SDF文件
    suppl = Chem.SDMolSupplier(input_sdf)
    molecules = [mol for mol in suppl if mol is not None]

    print(f"从文件 {input_sdf} 中读取了 {len(molecules)} 个有效分子")

    # 添加name属性
    writer = Chem.SDWriter(output_sdf)

    for idx, mol in tqdm(enumerate(molecules), total=len(molecules), desc="添加name属性"):
        mol_name = f"name{idx}"
        mol.SetProp("name", mol_name)
        writer.write(mol)

    writer.close()
    print(f"已将修改后的分子保存到 {output_sdf}")
if __name__ == "__main__":
    input_sdf = "A.sdf"        # 原始输入文件
    filtered_sdf = "B.sdf"    # 过滤后的临时文件
    test_sdf = "test.sdf"           # 用于保存前100个分子的文件
    final_sdf = "A.sdf"        # 最终添加name属性的输出文件（覆盖原始）

    # 步骤1: 过滤无效分子
    filter_sdf(input_sdf, filtered_sdf)

    # 步骤2: 分离前100个分子 -> test.sdf，剩下的 -> remain_mols
    suppl = Chem.SDMolSupplier(filtered_sdf)
    writer_test = Chem.SDWriter(test_sdf)
    remain_mols = []

    for idx, mol in enumerate(suppl):
        if mol is None:
            continue
        if idx < 200:
            mol_name = f"name{idx}"
            mol.SetProp("name", mol_name)
            writer_test.write(mol)
        else:
            remain_mols.append(mol)
    writer_test.close()
    print("前200个分子已保存至 test.sdf")

    # 步骤3: 为剩余分子添加 name 属性并写入最终文件
    writer_final = Chem.SDWriter(final_sdf)
    for idx, mol in enumerate(tqdm(remain_mols, desc="添加name属性")):
        mol.SetProp("name", f"name{idx}")
        writer_final.write(mol)
    writer_final.close()
    print(f"剩余分子已添加name属性并保存至 {final_sdf}")
    # add_name_property_to_sdf(test_sdf,test_sdf)