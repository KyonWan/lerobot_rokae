"""
测试十字手腕7轴机器人的正逆解
设置所有关节角度为10度，测试正解和逆解的一致性
"""

import math
import sys
import os

# 添加当前目录到路径，以便导入rokae_algo模块
# 假设编译后的.pyd文件在build/Release目录下
try:
    import rokae_algo
except ImportError:
    # 尝试从build目录导入
    build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'Release')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)
    try:
        import rokae_algo
    except ImportError:
        print("错误：无法导入rokae_algo模块")
        print("请确保已编译Python扩展模块，并且.pyd文件在Python路径中")
        sys.exit(1)


def test_cross_wrist7_ik():
    """测试十字手腕7轴机器人的正逆解"""
    
    print("=" * 60)
    print("测试十字手腕7轴机器人正逆解")
    print("=" * 60)
    
    # STEP 1: 初始化机器人参数
    # AR5-5_0.7R 机器人参数（24个杆长参数，单位：mm）
    # 格式：L01x, L01y, L01z, L12x, L12y, L12z, ..., L78x, L78y, L78z
    # 注意：Python接口传入的是毫米，内部会转换为米
    rob_dim = [
        0.0, 0.0, 0.0,      # L01
        0.0, 0.0, 174.5,    # L12
        0.0, 0.0, 314.0,    # L23
        10.0, 0.0, 0.0,     # L34
        -10.0, 0.0, 272.0,  # L45
        0.0, 0.0, 0.0,      # L56
        0.0, 0.0, 0.0,      # L67
        0.0, 0.0, 97.0      # L78
    ]

    for i in range(len(rob_dim)):
        rob_dim[i] = rob_dim[i] * 0.001
    
    # 关节角度限制（弧度）- 根据AR5-5_0.7R的实际参数
    min_joint = [
        math.radians(-178),  # 关节1
        math.radians(-120),  # 关节2
        math.radians(-178),  # 关节3
        math.radians(-60),   # 关节4
        math.radians(-178),  # 关节5
        math.radians(-110),  # 关节6
        math.radians(-180)   # 关节7
    ]
    max_joint = [
        math.radians(178),   # 关节1
        math.radians(120),   # 关节2
        math.radians(178),   # 关节3
        math.radians(145),   # 关节4
        math.radians(178),   # 关节5
        math.radians(110),   # 关节6
        math.radians(180)    # 关节7
    ]
    
    # 初始化机器人
    print("\n[步骤1] 初始化机器人...")
    res_init = rokae_algo.cross_wrist7_init(rob_dim, min_joint, max_joint)
    print(f"初始化结果: {res_init}")
    if not res_init:
        print("错误：机器人初始化失败！")
        return False
    
    # STEP 2: 设置所有关节角度为10度（转换为弧度）
    print("\n[步骤2] 设置初始关节角度为10度...")
    angle_deg = 10.0
    angle_rad = math.radians(angle_deg)
    init_joint_pos = [angle_rad] * 7
    print(f"初始关节角度（度）: {[math.degrees(q) for q in init_joint_pos]}")
    print(f"初始关节角度（弧度）: {init_joint_pos}")
    
    # STEP 3: 计算正解
    print("\n[步骤3] 计算正解（关节角度 -> 笛卡尔位姿）...")
    end_effector, psi, error_code = rokae_algo.cross_wrist7_jnt2cart(init_joint_pos)
    
    if error_code != 0:
        print(f"错误：正解计算失败，错误码: {error_code}")
        return False
    
    print(f"正解计算成功！")
    print(f"末端位置 (x, y, z): ({end_effector[0]:.6f}, {end_effector[1]:.6f}, {end_effector[2]:.6f})")
    print(f"末端姿态 (四元数 w, x, y, z): ({end_effector[3]:.6f}, {end_effector[4]:.6f}, {end_effector[5]:.6f}, {end_effector[6]:.6f})")
    print(f"臂角 psi (弧度): {psi:.6f}")
    print(f"臂角 psi (度): {math.degrees(psi):.6f}")
    
    # STEP 4: 使用逆解验证
    print("\n[步骤4] 计算逆解（笛卡尔位姿 -> 关节角度）...")
    out_joint_pos, error_code = rokae_algo.cross_wrist7_cart2jnt(init_joint_pos, end_effector, psi)
    
    if error_code != 0:
        print(f"错误：逆解计算失败，错误码: {error_code}")
        print("错误码说明:")
        print("  0: 成功")
        print("  1: 目标超出范围")
        print("  2: 输入数据错误")
        print("  3: 输入数据奇异")
        print("  4: 奇异")
        print("  5: 关节超限")
        print("  6: 步长过大")
        print("  7: 其他错误")
        return False
    
    print(f"逆解计算成功！")
    print(f"逆解关节角度（度）: {[math.degrees(q) for q in out_joint_pos]}")
    print(f"逆解关节角度（弧度）: {out_joint_pos}")
    
    # STEP 5: 验证正逆解一致性
    print("\n[步骤5] 验证正逆解一致性...")
    tolerance_rad = 1e-4  # 容差：约0.0057度
    max_error = 0.0
    all_match = True
    
    for i in range(7):
        error = abs(out_joint_pos[i] - init_joint_pos[i])
        # 考虑周期性，角度可能相差2π
        error = min(error, abs(error - 2*math.pi), abs(error + 2*math.pi))
        max_error = max(max_error, error)
        
        if error > tolerance_rad:
            all_match = False
            print(f"  关节 {i+1}: 误差 = {math.degrees(error):.6f}度 (超出容差)")
        else:
            print(f"  关节 {i+1}: 误差 = {math.degrees(error):.6f}度 ✓")
    
    print(f"\n最大误差: {math.degrees(max_error):.6f}度")
    
    if all_match:
        print("\n✓ 测试通过！正逆解一致性验证成功！")
    else:
        print("\n✗ 测试失败！部分关节角度误差超出容差")
    
    # STEP 6: 测试获取当前臂角
    print("\n[步骤6] 测试获取当前臂角...")
    cur_psi, success = rokae_algo.cross_wrist7_get_psi(init_joint_pos)
    if success:
        print(f"当前臂角 psi (弧度): {cur_psi:.6f}")
        print(f"当前臂角 psi (度): {math.degrees(cur_psi):.6f}")
        print(f"与正解返回的psi差异: {abs(cur_psi - psi):.6f} 弧度")
    else:
        print("获取臂角失败")
    
    # STEP 7: 测试奇异判断
    print("\n[步骤7] 测试奇异判断...")
    is_singular = rokae_algo.cross_wrist7_is_singular(init_joint_pos)
    print(f"是否奇异: {is_singular}")
    
    # STEP 8: 测试最近解接口
    print("\n[步骤8] 测试最近解接口...")
    out_joint_pos_nearest, error_code = rokae_algo.cross_wrist7_cart2jnt_nearest(
        init_joint_pos, end_effector, psi
    )
    if error_code == 0:
        print(f"最近解关节角度（度）: {[math.degrees(q) for q in out_joint_pos_nearest]}")
        # 计算与初始角度的距离
        distance = sum(abs(out_joint_pos_nearest[i] - init_joint_pos[i]) for i in range(7))
        print(f"与初始角度的总距离: {distance:.6f} 弧度")
    else:
        print(f"最近解计算失败，错误码: {error_code}")
    
    # 清理
    rokae_algo.de_init()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return all_match


if __name__ == "__main__":
    try:
        success = test_cross_wrist7_ik()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
