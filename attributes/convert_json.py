import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

class AttributeConverter:
    """属性词格式转换器"""
    
    @staticmethod
    def list_to_string(input_file: str, output_file: str, 
                      indent: str = '\t') -> None:
        """列表格式 → 字符串格式"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        attributes = data.get('attributes', {})
        converted = {k: ' '.join(v) for k, v in attributes.items()}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted, f, indent=indent, ensure_ascii=False)
        
        print(f"✓ {input_file} → {output_file}")
    
    @staticmethod
    def string_to_list(input_file: str, output_file: str,
                      dataset_name: str = "Unknown") -> None:
        """字符串格式 → 列表格式"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        attributes = {k: v.split() for k, v in data.items()}
        
        output_data = {
            "dataset": dataset_name,
            "num_classes": len(attributes),
            "words_per_class": len(next(iter(attributes.values()))) if attributes else 0,
            "attributes": attributes
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ {input_file} → {output_file}")
    
    @staticmethod
    def batch_convert(input_dir: str, output_dir: str, 
                     mode: str = 'list_to_string') -> None:
        """批量转换"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        json_files = list(input_path.glob("*.json"))
        
        for json_file in json_files:
            output_file = output_path / json_file.name
            
            try:
                if mode == 'list_to_string':
                    AttributeConverter.list_to_string(
                        str(json_file), str(output_file)
                    )
                else:
                    AttributeConverter.string_to_list(
                        str(json_file), str(output_file)
                    )
            except Exception as e:
                print(f"❌ {json_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='属性词格式转换工具')
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('output', help='输出文件或目录')
    parser.add_argument('--mode', choices=['list_to_string', 'string_to_list'],
                       default='list_to_string', help='转换模式')
    parser.add_argument('--batch', action='store_true', help='批量转换模式')
    parser.add_argument('--dataset', default='Unknown', help='数据集名称（反向转换时使用）')
    
    args = parser.parse_args()
    
    if args.batch:
        AttributeConverter.batch_convert(args.input, args.output, args.mode)
    else:
        if args.mode == 'list_to_string':
            AttributeConverter.list_to_string(args.input, args.output)
        else:
            AttributeConverter.string_to_list(args.input, args.output, args.dataset)


if __name__ == "__main__":
    main()  # ← 只保留这一行
