# TensorVision AI 错误报告与交接说明

项目目录: `C:\Users\skyi7\.gemini\antigravity\scratch\vision_ai_app`
Python: `E:\Python312\python.exe` (torch 2.6.0+cu124, ultralytics 8.4.x, PySide6 6.11)
GPU: RTX 4070 Laptop (sm_89, 8GB)

## 一、当前状态(已验证)
- exe 位置: `dist\TensorVisionAI\TensorVisionAI.exe`,启动约 2 秒显示窗口,界面响应正常,GPU 推理线程在运行。
- 离屏自检: 本地视频 -> 推理 -> 渲染全链路 29.8 FPS(视频原生 30 FPS)。
- 未验证: 真实摄像头画面、骨骼/车辆模式的肉眼效果、视频导出。

## 二、已修复的错误
| # | 文件 | 问题 | 后果 | 修复 |
|---|---|---|---|---|
| 1 | core/video_stream.py | 用了 `np.isnan` 但没 `import numpy` | 采集线程启动即崩,一帧都处理不了 | 改用 `math.isnan`,整个线程加 try/except |
| 2 | main.py | 窗口版 exe 的 `sys.stdout/stderr` 为 None | torch/ultralytics/tqdm 打印时可能崩,且无报错 | 重定向到 `%LOCALAPPDATA%\TensorVisionAI\app.log`,加 excepthook 弹窗 |
| 3 | run_app.bat | 含中文且 UTF-8 保存 | 中文 CMD(GBK)下命令被切成乱码 | 重写为纯 ASCII |
| 4 | ui/main_window.py | 推理在 GUI 线程,帧无限堆积 | 界面卡顿、延迟越来越大 | 新增 core/inference_worker.py,后台线程只处理最新帧 |
| 5 | main.py | 启动无反馈 | 导入 torch 时看起来像没打开 | 立即显示加载界面,并写阶段日志 |
| 6 | core/video_exporter.py | `finished = Signal(str)` 覆盖了 `QThread.finished` | 隐蔽 bug | 改名为 `done` |
| 7 | ui/main_window.py | 复选框用 `s == Qt.Checked` | 新版 PySide6 下不可靠 | 改用 `toggled(bool)` |
| 8 | core/video_stream.py | 停止线程时主线程释放 cap,且 stop 与 start 有竞态 | 摄像头占用、两个线程并存 | 由采集线程自己释放,start 前先置位 is_running |
| 9 | models/engine.py | 无线程锁 | 导出与实时预览并发会冲突 | 加 RLock(`_locked` 装饰器) |
| 10 | weights/ | 缺 yolov8s 权重 | 切换 Small 模型失败 | 已下载 4 个权重 |
| 11 | ui/video_widget.py, components.py | 短名枚举 | 兼容性隐患 | 改为完整枚举 |

## 三、遗留问题与注意事项
1. 重新打包后首次启动可能要等 1~2 分钟。原因是 Defender 实时扫描 `dist` 里几 GB 的新 DLL(日志栈停在 `torch._load_dll_libraries`),扫描完后恢复约 2 秒。可考虑把 `dist` 目录加入 Defender 排除项。
2. `main.py` 里还留着调试用的 `faulthandler.dump_traceback_later(25)` 和 `_t()` 阶段日志。启动正常时无影响,发布前可删。
3. 打包前必须先结束残留的 `TensorVisionAI.exe` 进程,否则 PyInstaller 因文件占用报 `PermissionError`。
4. `build_exe.py` 用 `--add-data` 复制了 weights 和源码目录,体积偏大(约 3GB+)。可以收窄:只带 weights,并 `--exclude-module tensorboard` 等。
5. 窗口版 exe 的 `MainWindowTitle` 读出为空,属于 Windows 取标题的现象,不代表窗口不存在。
6. 尚未实现: 摄像头分辨率与帧率设置、导出时的 GPU 占用提示、导出进度可取消后的半成品文件清理。

## 四、原始代码问题清单(给 Gemini 的教训)
- 只检查"进程是否存活"就宣称完成,从未真正喂过一帧。
- 写完后没有做过任何一次端到端运行(`test_pipeline.py` 只测了引擎,没测线程和界面)。
- 线程里没有任何异常处理,出错直接静默。
- 脚本文件含非 ASCII 字符,没考虑中文 Windows 的 GBK 编码。
- 没考虑 PyInstaller 窗口版没有 stdout 的情况。

## 五、验证方法(以后改完代码必做)
1. 离屏自检: 设 `QT_QPA_PLATFORM=offscreen`,创建 MainWindow,用 `capture_thread.open_file("demo_video.mp4")` 跑 6 秒,检查 `worker.fps > 0` 且日志里无 Traceback。
2. 打包后: 启动 exe,等到 `%LOCALAPPDATA%\TensorVisionAI\app.log` 出现 `window shown`。
3. 用 `nvidia-smi --query-compute-apps=name,used_memory --format=csv` 确认 exe 在用 GPU。
4. 不要只看进程存活;出现问题先看 `app.log`。

## 六、日志位置
`%LOCALAPPDATA%\TensorVisionAI\app.log`(即 `C:\Users\skyi7\AppData\Local\TensorVisionAI\app.log`)
