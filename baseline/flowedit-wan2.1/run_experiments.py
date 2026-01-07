#!/usr/bin/env python3
"""
统一的实验管理脚本
自动处理GPU选择、路径、批量运行等问题
"""

import os
import subprocess
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import time

# 实验配置 - 单一数据源
EXPERIMENTS = {
    "test01_watch_to_bracelet": {
        "video_path": "jewelry/JEWE001.mp4",
        "source_prompt": "A personalized ebony wood watch with engraved details, elegant men's wrist watch displayed on a white background.",
        "target_prompt": "A personalized engraved bar bracelet with curved nameplate, elegant slider bracelet displayed on a white background.",
        "source_blend": "watch",
        "target_blend": "bracelet",
    },
    "test02_tray_to_flowers": {
        "video_path": "home/HOME002.mp4",
        "source_prompt": "A molded leather catchall tray in vibrant colors, modern valet dish for jewelry and accessories on a white surface.",
        "target_prompt": "Fall vintage dried look hydrangeas in dusty pink, orange and brown autumn colors arranged as decorative centerpiece.",
        "source_blend": "tray",
        "target_blend": "hydrangeas",
    },
    "test03_stacker_to_ridetoy": {
        "video_path": "toys/TOYS001.mp4",
        "source_prompt": "Colorful wooden ring stacker with graduated circles, classic educational developmental toy for babies and toddlers.",
        "target_prompt": "Vintage style ride-on toy car in red, metal pedal car with chrome details for children.",
        "source_blend": "stacker",
        "target_blend": "toy",
    },
    "test04_socks_to_skirt": {
        "video_path": "clothing/CLOT001.mp4",
        "source_prompt": "Hand knitted wool socks, extra thick warm winter socks in neutral colors displayed on a clean background.",
        "target_prompt": "Midi wool plaid skirt, high waisted retro tartan swing skirt in autumn colors displayed elegantly.",
        "source_blend": "socks",
        "target_blend": "skirt",
    },
}

# 默认FlowAlign参数
DEFAULT_FLOWALIGN_PARAMS = {
    "strength": 0.7,
    "target_guidance_scale": 19.5,
    "flag_attnmask": True,
    "zeta_scale": 1e-3,
    "bg_zeta_scale": 1e-3,
}

# 默认推理参数
DEFAULT_INFERENCE_PARAMS = {
    "num_inference_step": 50,
}


class ExperimentRunner:
    def __init__(self, base_dir: Path, data_dir: Path, results_dir: Path, config_dir: Path):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.config_dir = config_dir

        # 确保目录存在
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def find_available_gpu(self) -> Optional[int]:
        """查找可用的GPU（内存使用<10GB）"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )

            for line in result.stdout.strip().split('\n'):
                gpu_id, mem_used = line.split(',')
                gpu_id = int(gpu_id.strip())
                mem_used = int(mem_used.strip())

                # 如果GPU内存使用<10GB，认为可用
                if mem_used < 10000:
                    print(f"[GPU] Found available GPU {gpu_id} (mem used: {mem_used}MB)")
                    return gpu_id

            print("[GPU] Warning: No GPU with <10GB memory usage found, will use GPU 0")
            return 0

        except Exception as e:
            print(f"[GPU] Error checking GPU status: {e}, will use GPU 0")
            return 0

    def create_config(self, exp_name: str, exp_config: Dict) -> Path:
        """创建实验配置文件"""
        config_path = self.config_dir / f"{exp_name}.yaml"

        # 构建完整视频路径
        video_full_path = self.data_dir / "pvtt-benchmark" / "videos" / exp_config["video_path"]

        config = {
            "video": {
                "video_path": str(video_full_path),
                "source_prompt": exp_config["source_prompt"],
                "target_prompt": exp_config["target_prompt"],
                "source_blend": exp_config["source_blend"],
                "target_blend": exp_config["target_blend"],
            },
            "infernece": DEFAULT_INFERENCE_PARAMS,
            "training-free-type": {
                "flag_flowedit": False,
                "flag_flowalign": True,
            },
            "flowedit": {
                "strength": 0.7,
                "target_guidance_scale": 13.5,
                "source_guidance_scale": 5.0,
                "save_video": str(self.results_dir / f"{exp_name}_flowedit.mp4"),
            },
            "flowalign": {
                **DEFAULT_FLOWALIGN_PARAMS,
                "save_video": str(self.results_dir / f"{exp_name}_flowalign.mp4"),
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"[Config] Created: {config_path}")
        return config_path

    def run_experiment(self, exp_name: str, exp_config: Dict, gpu_id: Optional[int] = None) -> bool:
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")

        # 查找可用GPU
        if gpu_id is None:
            gpu_id = self.find_available_gpu()

        # 创建配置文件
        config_path = self.create_config(exp_name, exp_config)

        # 构建命令
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['HF_ENDPOINT'] = 'https://hf-mirror.com'

        cmd = [
            "python",
            "awesome_wan_editing.py",
            f"--config={config_path}"
        ]

        print(f"[Command] CUDA_VISIBLE_DEVICES={gpu_id} python awesome_wan_editing.py --config={config_path}")

        # 运行实验
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True
            )

            print(f"[Success] {exp_name} completed!")
            print(f"[Output] {result.stdout[-500:]}")  # 打印最后500字符
            return True

        except subprocess.CalledProcessError as e:
            print(f"[Error] {exp_name} failed!")
            print(f"[STDOUT] {e.stdout[-1000:]}")
            print(f"[STDERR] {e.stderr[-1000:]}")
            return False

    def run_all(self, experiments: Optional[List[str]] = None, sequential: bool = False):
        """运行所有或指定的实验"""
        exp_list = experiments if experiments else list(EXPERIMENTS.keys())

        print(f"[Batch] Will run {len(exp_list)} experiments")
        print(f"[Mode] {'Sequential' if sequential else 'Parallel'}")

        if sequential:
            # 顺序运行
            results = {}
            for exp_name in exp_list:
                exp_config = EXPERIMENTS[exp_name]
                success = self.run_experiment(exp_name, exp_config)
                results[exp_name] = success

                if not success:
                    print(f"[Warning] {exp_name} failed, continuing to next experiment...")
        else:
            # 并行运行（简单实现：为每个实验分配不同GPU）
            print("[Parallel] Launching all experiments in background...")
            processes = []

            for i, exp_name in enumerate(exp_list):
                exp_config = EXPERIMENTS[exp_name]
                gpu_id = self.find_available_gpu()

                # 创建配置
                config_path = self.create_config(exp_name, exp_config)

                # 启动进程
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                env['HF_ENDPOINT'] = 'https://hf-mirror.com'

                proc = subprocess.Popen(
                    ["python", "awesome_wan_editing.py", f"--config={config_path}"],
                    cwd=self.base_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                processes.append((exp_name, proc))
                print(f"[Parallel] Launched {exp_name} on GPU {gpu_id}")
                time.sleep(2)  # 避免同时启动导致冲突

            # 等待所有进程完成
            print(f"[Parallel] Waiting for {len(processes)} experiments to complete...")
            results = {}
            for exp_name, proc in processes:
                proc.wait()
                results[exp_name] = proc.returncode == 0
                status = "✓" if results[exp_name] else "✗"
                print(f"[{status}] {exp_name}")

        # 打印总结
        print(f"\n{'='*60}")
        print("Experiment Results Summary")
        print(f"{'='*60}")
        for exp_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status:12} {exp_name}")


def main():
    parser = argparse.ArgumentParser(description="统一的PVTT实验管理脚本")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="指定要运行的实验，不指定则运行全部"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="顺序运行实验（默认并行）"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="手动指定GPU ID（仅在顺序模式下有效）"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="FlowEdit-WAN2.1 代码目录"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data",
        help="数据目录（包含pvtt-benchmark）"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="结果保存目录（默认：项目根目录/experiments/results/flowalign-wan2.1）"
    )

    args = parser.parse_args()

    # 默认结果目录：pvtt/experiments/results/flowalign-wan2.1/
    if args.results_dir is None:
        args.results_dir = args.base_dir.parent.parent / "experiments" / "results" / "flowalign-wan2.1"

    # 初始化Runner
    runner = ExperimentRunner(
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        config_dir=args.base_dir / "config" / "pvtt"
    )

    # 运行实验
    runner.run_all(
        experiments=args.experiments,
        sequential=args.sequential
    )


if __name__ == "__main__":
    main()
