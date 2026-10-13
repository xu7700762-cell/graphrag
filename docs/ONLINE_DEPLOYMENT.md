# 实时在线分析与树莓派 5 部署

新增入口为 `/online`，与原始离线分析共存。在线回放读取同一批 CDT/DPO，默认 1 倍速、256 个原始样本一块，每五秒显示质量、VRMS 前缀读数和疲劳/工作负荷 EEG 指标。完整 `20→22` 路径结束后才发布整路径结果并按选项请求云端。过程读数不等于即时症状真值，原离线准确率也不能直接转为新 CPU FP32 协议的准确率。

在线信号合同固定为 1024 Hz、指定 30 个 EEG 通道、M1/M2 平均参考、50 Hz 陷波、0.5–45 Hz 四阶带通、70 Hz 八阶低通、有状态因果滤波、五秒预热与全局 4 倍降采样；没有记录开头静息参考，也没有新增眼电回归。`pi_online.yaml` 中旧监测合同的 `warmup_seconds: 10` 仅用于旧配置加载，在线模块独立使用固定五秒预热。

## 已完成与待实机验证

- 26 份原始 CDT、51,823.6875 秒信号、9,545 个任务/休息窗口已与当前 D2 离线预处理逐值核对；数组含 NaN 的比较、质量门控、事件边界、序号、完整性和窗口数均一致，最大数值误差为零。这是加速数值检查。
- CPU 后端使用普通 PyTorch 的 mamba-minimal FP32 移植，完整加载原 83 个编码器张量和冻结 MIL/八工具参数；迁移误差和 BF16/FP32 边界见 `ONLINE_MODEL.md` 及包内 `validation/`。
- 已实际下载并检查 ARM64 CPython 3.11 二进制依赖闭包，独立 wheel 缓存不安装到 Windows 环境。下载成功证明轮子可用；ARM64 导入、1 倍速延迟、内存、温度、降频和两小时稳定性仍需树莓派实测。
- 24 个留一折覆盖 1–21、23、24、25；22/26 没有对应权重，描述性工具可继续，VRMS 分类明确不可用。

本轮尚未连接树莓派，也没有树莓派硬件性能通过报告。Windows/WSL 的耗时仅是开发验证。实际云端回复可能变化；本地工程验收默认关闭云端，不花费模型 API 调用。

## 交付包与依赖

在 Windows 项目目录构建或刷新新包：

```powershell
.\.venv\Scripts\python.exe scripts/build_online_delivery.py --check-wheels
```

输出为 `deploy/EEGAgent_Online_Pi_bundle.zip`，外部清单为 `deploy/online_bundle_manifest.json`。构建器按白名单收集源码、在线网页、内置知识卡、部署配置、冻结可移植资产、ARM64 wheels 和必要验证脚本，逐文件校验压缩包 SHA256/CRC。旧部署包和历史输出不修改。新包含本地私有冻结参数，不含原始数据、标签、`.env`、`cloud_profiles.json`、API 响应、历史实验输出或 EEG 缓存数组。

目标固定为 Raspberry Pi OS **64 位 Bookworm / CPython 3.11**，glibc 至少 2.28。主依赖固定为 Torch 2.7.1、NumPy 2.0.2、SciPy 1.13.1、scikit-learn 1.6.1、joblib 1.5.3；其余版本和全部传递依赖见 `requirements-online-pi.lock.txt`。NumPy 2.0.2 与已有 NumPy 2 序列化资产匹配。安装仅接受 wheel，不编译 CUDA、mamba-ssm 或训练扩展，也不使用 Windows 的虚拟环境。

将包解压到 `/srv/eeg-agent-online`，另行将自己的原始数据放到：

```text
/srv/eeg-data/data/raw/Acquisition 01.cdt
/srv/eeg-data/data/raw/Acquisition 01.cdt.dpo
...
```

在线入口不需要 CEO 或标签 CSV。模型清单为 `/srv/eeg-agent-online/outputs/online_assets/seed2026/manifest.json`；保留整个资产目录及文件相对位置，不能只复制编码器。

在树莓派先检查平台，再安装：

```bash
uname -m                    # 必须为 aarch64
python3.11 --version        # 必须使用 3.11
getconf GNU_LIBC_VERSION
sudo apt-get install python3.11-venv libgomp1
sudo python3.11 -m venv /srv/eeg-agent-online/.venv
cd /srv/eeg-agent-online
sudo .venv/bin/python -m pip install --no-index --only-binary=:all: \
  --find-links wheels/arm64_cp311 --require-hashes -r requirements-online-pi.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import torch,numpy,scipy,sklearn,onnxruntime; print(torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, sklearn.__version__)"
```

`torch.version.cuda` 应为 `None`。如设备使用 Python 3.13、32 位系统或不匹配的 libc，先选择符合目标的平台；不要把 pip 自动编译或未经验证的替换版本当作安装成功。

解压后可核验包内每个文件：

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
for row in json.loads(Path('deploy/online_bundle_manifest.json').read_text())['entries']:
    actual = hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()
    assert actual == row['sha256'], row['file']
print('Bundle hashes verified')
PY
```

## 服务与访问

使用独立服务账户，程序、数据和资产只读；运行会话、知识索引和本项目云端配置写入 `/var/lib/eeg-agent-online`：

```bash
if ! id -u eegagent >/dev/null 2>&1; then
  sudo useradd --system --home-dir /var/lib/eeg-agent-online --shell /usr/sbin/nologin eegagent
fi
sudo install -d -o eegagent -g eegagent -m 750 /var/lib/eeg-agent-online
sudo chown -R root:eegagent /srv/eeg-agent-online
sudo chmod -R g+rX,g-w /srv/eeg-agent-online
sudo -u eegagent test -r '/srv/eeg-data/data/raw/Acquisition 01.cdt'
sudo install -m 644 deploy/eeg-agent-online.service /etc/systemd/system/eeg-agent-online.service
sudo install -m 644 deploy/eeg-agent-online-journald.conf /etc/systemd/journald@eeg-agent-online.conf
sudo install -m 644 deploy/eeg-agent-online.logrotate /etc/logrotate.d/eeg-agent-online
sudo systemctl daemon-reload
sudo systemctl enable --now eeg-agent-online.service
sudo systemctl status eeg-agent-online.service
```

服务使用一个 Uvicorn 工作进程、CPU 两线程，监听 `0.0.0.0:8880`，失败后重启。先确认端口 8880 没有被另一服务占用。浏览器打开 `http://树莓派IP:8880/online`；云端设置入口为 `/monitor#cloud`。云端配置保存在 `/var/lib/eeg-agent-online/.env`，不把桌面项目中的密钥文件复制进程序目录。

查看独立服务日志：

```bash
sudo journalctl --namespace=eeg-agent-online -u eeg-agent-online.service -f
sudo logrotate -d /etc/logrotate.d/eeg-agent-online
```

journal 独立预算为 256 MB / 7 天。每会话显示事件由程序按 64 MB 保留当前文件和一个旧文件；不含波形的 `evidence.jsonl` 由程序保留 64 MB×3 段。外部 logrotate 仅轮转基准遥测文件，不干预会话事件的内置轮转。快照和路径报告不轮转。这些是每日志/每会话预算，不是整个状态目录的总磁盘配额。服务重启会把在途任务标为中断；新建会话重新回放及预热，不从中断的滤波状态继续。

## 树莓派实测验收

首次运行完整 Acquisition 01 的真实 1 倍速测量：

```bash
cd /srv/eeg-agent-online
sudo -u eegagent .venv/bin/python scripts/benchmark_online_pi.py
# 完整路径与CPU环境检查通过后，再运行两小时重复回放
sudo -u eegagent .venv/bin/python scripts/benchmark_online_pi.py --hours2
```

脚本通过本机在线 API 启动 `speed=1, cloud=false`，用 SSE 获取每窗实际分析时间和完成延迟，每秒测量服务 RSS、CPU、队列、温度及 `vcgencmd get_throttled`。默认报告写入 `/var/lib/eeg-agent-online/benchmarks/`，保留原始无波形遥测。两小时模式重复建立新会话，达到两小时后停止最后一条回放。

默认完整回放须以 `completed` 结束、读到实际文件 EOF，原始样本数和位置均与目录一致；`stopped` 不算通过。两小时模式仅允许脚本达到截止时间后主动停止仍在运行的最后一个会话，外部提前停止判失败。报告保存逐会话 `acceptance_checks` 和停止来源，读取端生成数、推理完成数与 SSE 收到窗口数须全部一致。

验收还核对目录中的 CPU 后端、返回会话的实际 `speed=1`、`replay_mode=realtime`、服务测量主机 `aarch64` 及 `cloud.requested=false`，不以提交的请求参数代替实际执行证据。报告保存资源采样间隔与超过 2.5 秒的空档（含开始、采样执行和结束阶段）；即使已有样本字段齐全，存在超限空档也不能判定通过。

验收报告分别给出 P50/P95/max；每窗处理及完成延迟 P95 须小于五秒、分析队列须小于八事件、服务 RSS 目标不超过 4 GB、整机已用内存目标不超过 6 GB，没有过载/降频或缺失遥测，并且实际产生模型读数。SSE 窗口计数与服务累计计数核对，缺失事件不能判定通过。`vcgencmd` 或温度不可用时保留未测，不能判定稳定性通过。两小时模式要求实际采样达到 7200 秒；故障注入另行验收。电功耗需要外接电表，脚本不会从 CPU 利用率推算。非树莓派 5 / 非 ARM64 主机只生成 `hardware_not_tested`，退出非零，不请求回放。

报告 schema 为 `pi5_online_benchmark_v2`，逐项保存资源缺样数量及比例。即使前面的温度/降频正常，后续缺样也不能判定通过。此版本须配合同包更新后的在线服务，旧服务缺少采流端窗口数或 EOF 证据时，验收会明确失败。

包内包含在线测试及其共享的合成数据辅助文件。开发端会在独立解压目录运行这些测试，并核验实际导入来自包内源码；这验证交付包的完整性，不代表 ARM64 依赖已在树莓派导入或通过实机验收。

之后可在网页单独启用路径结束云端，检查断网/超时保留本地结果、云端失败不代替最终云端类别、暂停/继续/停止、刷新与 SSE 恢复；云端阶段的网络等待另计，不能混入本地五秒窗口延迟。科学效果评价仍按固定被试隔离和路径标签事后评分，前缀曲线没有逐时症状真值支持。
