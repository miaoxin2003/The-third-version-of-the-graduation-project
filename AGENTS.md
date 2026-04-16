## Python视觉端 (svm/)
- 依赖: pip install opencv-python mediapipe scikit-learn joblib pyserial pandas numpy
- 采集数据: python svm/collect_data.py
- 训练模型: python svm/train_model.py  # 用dataset.csv生成gesture_model.pkl
- 运行跟踪: python svm/HandTrackingModule.py  # 默认COM8@115200发送#x$y\r\n (640x480,中心320x240)
- Demo: python svm/inference.py 或 svm/FingerCounter.py

## STM32控制端 (control/)
- Keil uVision: 编译CONTROL项目，烧录OBJ/CONTROL.hex
- 串口: uart_init(115200)，接收#x$y\r\n → coords[0/1]
- PID: pid_init(0.04,0,0.30,&PID_x); pid_init(0.05,0,0.30,&PID_y); 输出限幅300-1200
- PWM: TIM3_PWM_Init(9999,143) → CH1/2@50Hz
- 清理: keilkilll.bat

无测试/lint/git。坐标范围~0-640/0-480，目标320/240。