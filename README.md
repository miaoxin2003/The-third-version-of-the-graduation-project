# 手势控制云台系统 (Gesture-Controlled Pan-Tilt System)

基于视觉手势识别与PID控制的云台跟随系统，适用于电气自动化专业毕业论文。

## 系统架构

```
┌─────────────────┐     #x$y\r\n      ┌─────────────────┐
│  PC视觉端       │  ──── 115200 ────▶ │   STM32控制端    │
│  (Python/PC)    │   COM8 Serial      │   (嵌入式)       │
│  MediaPipe+FPID │                    │   PID+PWM        │
└─────────────────┘                    └─────────────────┘
       │                                      │
       ▼                                      ▼
  手势识别                          云台双轴跟随
  (SVM)                             (TIM3 CH1/2)
```

## 项目结构

```
.
├── svm/                          # Python视觉端
│   ├── HandTrackingModule.py     # 核心：手部检测+EMA滤波+预测+FPID
│   ├── FuzzyPID.py               # 模糊PID控制器 (3级规则, O(1)推理)
│   ├── collect_data.py           # 手势数据采集
│   ├── train_model.py           # SVM模型训练
│   ├── inference.py             # 手势识别Demo
│   ├── FingerCounter.py         # 手指计数Demo
│   ├── fuzzy_table.npy           # 预计算模糊规则表
│   └── gesture_model.pkl        # 训练好的SVM模型
│
├── control/                      # STM32控制端 (Keil uVision)
│   ├── USER/main.c              # 主程序：串口接收+PID+PWM
│   ├── HARDWARE/
│   │   ├── PID/pid.c/h         # 位置式/增量式PID控制器
│   │   ├── serial/serial.c/h   # 串口协议解析 (#x$y\r\n)
│   │   ├── TIMER/timer.c/h     # TIM3 PWM初始化 (50Hz)
│   │   ├── LED/led.c/h         # LED指示
│   │   └── KEY/key.c/h         # 按键扫描
│   ├── CORE/                    # CMSIS核心文件
│   └── OBJ/CONTROL.hex          # 编译产物 (可烧录)
│
├── dataset.csv                  # 手势数据集
├── gesture_model.pkl            # 训练模型
└── AGENTS.md                    # 开发者指令
```

## 硬件清单

| 组件 | 规格 | 说明 |
|------|------|------|
| PC摄像头 | 640x480 | 手势采集 |
| STM32F103 | 72MHz | 主控芯片 |
| 云台舵机 | 50Hz PWM 300-1200μs | 双轴跟随 |
| 串口 | COM8 @ 115200 | PC↔STM32通信 |

## 依赖安装

```bash
pip install opencv-python mediapipe scikit-learn joblib pyserial pandas numpy scipy matplotlib
```

## 快速运行

### 1. 数据采集
```bash
python svm/collect_data.py
# 按's'保存坐标，按'q'退出
```

### 2. 训练模型
```bash
python svm/train_model.py
# 生成 gesture_model.pkl
```

### 3. 运行主程序
```bash
python svm/HandTrackingModule.py
# 按's'退出
```

## 创新点

1. **自适应EMA滤波+速度预测** - 手部坐标低延迟防抖 (<20ms)
2. **SVM手势识别** - MediaPipe 21关键点特征提取
3. **模糊PID控制器 (FPID)** - 3级规则动态调参，超调减25%
4. **双轴PID+积分限幅** - 防积分饱和，稳定跟随
5. **手势状态机** - WAITING→CONFIRMING→TRIGGERED防误触

详细分析见 [创新点.md](创新点.md)

## 通信协议

**PC → STM32** (115200 baud):
```
#x$y\r\n   # x,y为整数坐标 (0-640, 0-480)
#xKp:0.045#xKd:0.320#xU:450#yKp:0.050#yKd:0.310#yU:650#  # FPID参数(可选)
```

**STM32 → PC** (USB调试):
```
Received: #320$240\r\n
Parsed Coords: X=320, Y=240
PWM Values: X=650, Y=650
```

## STM32固件烧录

1. 打开Keil uVision，加载 `control/CONTROL.uvproj`
2. 编译后烧录 `control/OBJ/CONTROL.hex`

或使用J-Link/ST-Link直接烧录hex文件

## 性能指标

| 指标 | 数值 |
|------|------|
| 跟踪精度 | ±5px |
| 响应延迟 | <50ms |
| FPS | 30+ |
| 超调量 | <25% |

## 作者

河南理工大学 电气自动化专业

## 开源协议

MIT License
