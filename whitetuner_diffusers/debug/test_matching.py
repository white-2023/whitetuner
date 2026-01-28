import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_base_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    if '_' in stem:
        parts = stem.rsplit('_', 1)
        if parts[1].isdigit():
            return parts[0]
    return stem

def match_condition(target_name: str, condition_file: str) -> bool:
    condition_stem = os.path.splitext(condition_file)[0]
    condition_base_id = get_base_id(condition_file)
    return target_name == condition_stem or target_name == condition_base_id

def test_get_base_id():
    print("=" * 60)
    print("测试 get_base_id 函数")
    print("=" * 60)
    
    test_cases = [
        ("001.png", "001"),
        ("001_1.png", "001"),
        ("001_1_0.png", "001_1"),
        ("001_1_1.png", "001_1"),
        ("001_1_2.png", "001_1"),
        ("abc.jpg", "abc"),
        ("abc_0.jpg", "abc"),
        ("abc_def.jpg", "abc_def"),
        ("abc_def_1.jpg", "abc_def"),
        ("image.png", "image"),
        ("image_001.png", "image"),
        ("a_b_c_1.png", "a_b_c"),
        ("test_image_02.jpg", "test_image"),
        ("no_number_suffix_x.png", "no_number_suffix_x"),
        ("0001_1.png", "0001"),
    ]
    
    all_passed = True
    for filename, expected in test_cases:
        result = get_base_id(filename)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: get_base_id('{filename}') = '{result}' (expected: '{expected}')")
    
    return all_passed

def test_matching():
    print("\n" + "=" * 60)
    print("测试匹配逻辑")
    print("=" * 60)
    
    test_cases = [
        ("001_1", "001_1.png", True, "完全同名"),
        ("001_1", "001_1_0.png", True, "target + _0 后缀"),
        ("001_1", "001_1_1.png", True, "target + _1 后缀"),
        ("001_1", "001_1_2.png", True, "target + _2 后缀"),
        ("001_1", "001_2.png", False, "不同编号"),
        ("001_1", "001.png", False, "target 比 condition 长"),
        ("001", "001.png", True, "无后缀完全匹配"),
        ("001", "001_0.png", True, "无后缀 target 匹配带后缀 condition"),
        ("001", "001_1.png", True, "无后缀 target 匹配带后缀 condition"),
        ("abc", "abc.jpg", True, "字母名称完全匹配"),
        ("abc", "abc_0.jpg", True, "字母名称匹配带后缀"),
        ("abc", "abc_def.jpg", False, "后缀不是数字"),
        ("abc_def", "abc_def.jpg", True, "多下划线完全匹配"),
        ("abc_def", "abc_def_1.jpg", True, "多下划线匹配带后缀"),
        ("abc_def", "abc.jpg", False, "target 比 condition 长"),
        ("test_image", "test_image.png", True, "下划线名称完全匹配"),
        ("test_image", "test_image_01.png", True, "下划线名称匹配带后缀"),
        ("test_image", "test_image_02.png", True, "下划线名称匹配带后缀2"),
        ("0001_1", "0001_1.png", True, "用户实际场景: 完全同名"),
        ("0001_1", "0001_1_0.png", True, "用户实际场景: 带后缀0"),
        ("0001_1", "0001_1_1.png", True, "用户实际场景: 带后缀1"),
        ("0001_1", "0001_2.png", False, "用户实际场景: 不同编号不匹配"),
        ("0001_1", "0001.png", False, "用户实际场景: 短名称不匹配"),
    ]
    
    all_passed = True
    for target_name, condition_file, expected, desc in test_cases:
        result = match_condition(target_name, condition_file)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        result_str = "匹配" if result else "不匹配"
        expected_str = "匹配" if expected else "不匹配"
        print(f"  {status}: target='{target_name}' condition='{condition_file}' -> {result_str} (expected: {expected_str}) [{desc}]")
    
    return all_passed

def test_batch_matching():
    print("\n" + "=" * 60)
    print("测试批量匹配场景")
    print("=" * 60)
    
    target_files = ["0001_1.png", "0002_1.png", "abc.jpg", "test_image.png"]
    condition_files = [
        "0001_1.png", "0001_1_0.png", "0001_1_1.png",
        "0002_1.png", "0002_1_0.png",
        "abc.jpg", "abc_0.jpg", "abc_1.jpg",
        "test_image.png", "test_image_01.png",
        "unmatched.png", "random_file.jpg"
    ]
    
    print(f"\nTarget 文件: {target_files}")
    print(f"Condition 文件: {condition_files}")
    print()
    
    for target_file in target_files:
        target_name = os.path.splitext(target_file)[0]
        matched = []
        for cond_file in condition_files:
            if match_condition(target_name, cond_file):
                matched.append(cond_file)
        print(f"  Target '{target_file}' (name='{target_name}') 匹配到: {matched}")

def main():
    print("图片匹配逻辑测试\n")
    
    result1 = test_get_base_id()
    result2 = test_matching()
    test_batch_matching()
    
    print("\n" + "=" * 60)
    if result1 and result2:
        print("所有测试通过!")
    else:
        print("存在失败的测试!")
    print("=" * 60)

if __name__ == "__main__":
    main()

