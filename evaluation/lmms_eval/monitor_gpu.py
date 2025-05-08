import time, pynvml, multiprocessing as mp

def gpu_monitor(shared_dict,
                interval: float = 0.5,
                mode: str = "sum"):
    """
    mode 설명
    ----------
    "sum" : 모든 GPU 메모리 총합 (device_map='auto' → 계층적 샤딩)
    "max" : 모든 GPU에 올라간 모든 PID 중에서
            VRAM을 가장 많이 쓰는 '단일 프로세스'의 사용량
    """
    import pynvml, time, collections

    pynvml.nvmlInit()
    dev_cnt = pynvml.nvmlDeviceGetCount()
    peak_val = 0.0
    # 버전에 따라 *_v2/_v3가 존재할 수 있으므로 안전하게 래핑
    def _get_proc_infos(handle):
        for fn_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses_v3",
        ):
            if hasattr(pynvml, fn_name):
                return getattr(pynvml, fn_name)(handle)
        return []

    try:
        while shared_dict["run"]:
            if mode == "sum":
                # ── 기존: GPU 단위 총합 ───────────────────────────
                usages = []
                for i in range(dev_cnt):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**3
                    usages.append(0.0 if mem < 1.0 else mem)
                current = sum(usages)

            else:  # mode == "max" : 프로세스 단위 최대
                max_proc_mem = 0.0
                for i in range(dev_cnt):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    # compute context
                    for p in _get_proc_infos(h):
                        mem_gb = p.usedGpuMemory / 1024**3
                        max_proc_mem = max(max_proc_mem, mem_gb)
                    # 그래픽(context)까지 보려면 아래 주석 해제
                    # for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
                    #     mem_gb = p.usedGpuMemory / 1024**3
                    #     max_proc_mem = max(max_proc_mem, mem_gb)

                current = max_proc_mem

            peak_val = max(peak_val, current)
            shared_dict["peak"] = peak_val
            time.sleep(interval)

    finally:
        pynvml.nvmlShutdown()
