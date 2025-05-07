import time, pynvml, multiprocessing as mp

def gpu_monitor(shared_dict, interval=0.5, mode="sum"):
    """
    mode = "sum" : 모든 GPU 사용량 합계를 모니터 (멀티-GPU 샤딩)
    mode = "max" : GPU별 사용량 중 최대값을 모니터 (단일-GPU 실행)
    """
    pynvml.nvmlInit()
    dev_cnt = pynvml.nvmlDeviceGetCount()
    peak = 0.0

    try:
        while shared_dict["run"]:
            usages = []
            for i in range(dev_cnt):
                handle  = pynvml.nvmlDeviceGetHandleByIndex(i)
                used_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
                used_gb = 0.0 if used_gb < 1.0 else used_gb   # 1 GB 미만 무시
                usages.append(used_gb)

            current = sum(usages) if mode == "sum" else max(usages)
            peak = max(peak, current)
            shared_dict["peak"] = peak        # 실시간 최대값 공유
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()
