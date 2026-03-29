import argparse

# 模拟命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--out_channels", type=int)

# 模拟命令行输入
args = parser.parse_args(["--out_channels", "38"])
print(f"out_channels: {args.out_channels}")
print(f"Type: {type(args.out_channels)}")
