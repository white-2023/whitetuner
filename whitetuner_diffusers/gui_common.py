import sys
import io
import os

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

import threading
from queue import Queue
import time
import re
import subprocess
import socket
import signal
import atexit

TKINTER_AVAILABLE = False
try:
    from tkinter import Tk, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    pass


def detect_gui_environment():
    if os.name != 'nt':
        display = os.environ.get('DISPLAY')
        if not display:
            return False
    
    if TKINTER_AVAILABLE:
        try:
            root = Tk()
            root.withdraw()
            root.destroy()
            return True
        except Exception:
            return False
    
    return False


training_process = None
training_child_pids = []
is_training = False
should_stop = False
is_local_mode = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_all_child_pids(parent_pid):
    pids = []
    try:
        import psutil
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            pids.append(child.pid)
    except:
        pass
    return pids


def kill_all_training_processes():
    global training_child_pids, training_process
    
    pids_to_kill = list(training_child_pids)
    
    if training_process is not None:
        try:
            pids_to_kill.extend(get_all_child_pids(training_process.pid))
            pids_to_kill.append(training_process.pid)
        except:
            pass
    
    pids_to_kill = list(set(pids_to_kill))
    
    if not pids_to_kill:
        return
    
    print(f"[kill] 正在强制终止 {len(pids_to_kill)} 个进程: {pids_to_kill}")
    
    for pid in pids_to_kill:
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
        except:
            pass
    
    training_child_pids.clear()

log_queue = Queue()
original_stdout = sys.stdout
original_stderr = sys.stderr

loss_log_buffer = []
LOSS_LOG_MAX_LINES = 100000
_log_buffer_lock = threading.Lock()


def append_to_loss_log(text: str):
    global loss_log_buffer
    if text:
        with _log_buffer_lock:
            lines = text.split('\n')
            loss_log_buffer.extend(lines)
            if len(loss_log_buffer) > LOSS_LOG_MAX_LINES:
                loss_log_buffer = loss_log_buffer[-LOSS_LOG_MAX_LINES:]


def get_loss_log() -> str:
    with _log_buffer_lock:
        return '\n'.join(loss_log_buffer)


def clear_loss_log():
    global loss_log_buffer
    with _log_buffer_lock:
        loss_log_buffer = []


def run_training_process(cmd, cwd=None, env=None, initial_msg=""):
    global training_process, is_training, should_stop, training_child_pids
    
    if is_training:
        yield "[X] 训练正在进行中，请等待当前训练完成"
        return
    
    should_stop = False
    training_child_pids.clear()
    clear_loss_log()
    
    while not log_queue.empty():
        log_queue.get()
    
    accumulated_output = initial_msg
    if initial_msg:
        append_to_loss_log(initial_msg)
    yield accumulated_output
    
    try:
        if env is None:
            env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.STDOUT,
            'bufsize': 1,
            'cwd': cwd or SCRIPT_DIR,
            'env': env,
            'encoding': 'utf-8',
            'errors': 'replace',
        }
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs['start_new_session'] = True
        
        training_process = subprocess.Popen(cmd, **popen_kwargs)
        is_training = True
        
        def monitor_child_pids():
            while is_training and training_process is not None:
                try:
                    new_pids = get_all_child_pids(training_process.pid)
                    for pid in new_pids:
                        if pid not in training_child_pids:
                            training_child_pids.append(pid)
                except:
                    pass
                time.sleep(2)
        
        pid_monitor_thread = threading.Thread(target=monitor_child_pids, daemon=True)
        pid_monitor_thread.start()
        
        def read_stdout():
            try:
                for line in iter(training_process.stdout.readline, ''):
                    if not line:
                        break
                    log_queue.put(line)
                    append_to_loss_log(line)
                    try:
                        original_stdout.write(line)
                        original_stdout.flush()
                    except:
                        pass
            except Exception as e:
                log_queue.put(f"[X] 读取输出错误: {str(e)}\n")
            finally:
                training_process.stdout.close()
        
        reader_thread = threading.Thread(target=read_stdout, daemon=True)
        reader_thread.start()
        
        last_yield_time = 0
        yield_interval = 0.1
        
        while training_process.poll() is None:
            current_time = time.time()
            
            new_logs = []
            while not log_queue.empty():
                try:
                    log_item = log_queue.get_nowait()
                    new_logs.append(log_item)
                except:
                    break
            
            if new_logs:
                new_content = ''.join(new_logs)
                new_content = re.sub(r'\n{3,}', '\n\n', new_content)
                accumulated_output += new_content
                if len(accumulated_output) > 50000:
                    accumulated_output = accumulated_output[-40000:]
            
            if current_time - last_yield_time >= yield_interval:
                yield accumulated_output
                last_yield_time = current_time
            else:
                time.sleep(0.05)
        
        reader_thread.join(timeout=2.0)
        
        while not log_queue.empty():
            try:
                log_item = log_queue.get_nowait()
                accumulated_output += log_item
            except:
                break
        
        return_code = training_process.returncode
        is_training = False
        training_process = None
        
        if should_stop:
            stop_status = "\n\n[ok] 训练已停止，模型和检查点已保存"
        elif return_code == 0:
            stop_status = "\n\n[ok] 训练完成!"
        else:
            stop_status = f"\n\n[X] 训练进程异常退出 (return code: {return_code})"
        
        accumulated_output += stop_status
        append_to_loss_log(stop_status)
        yield accumulated_output
        
    except Exception as e:
        is_training = False
        training_process = None
        error_msg = f"\n\n[X] 训练启动失败: {str(e)}"
        append_to_loss_log(error_msg)
        yield f"{accumulated_output}{error_msg}"

tensorboard_process = None
tensorboard_port = 6006
tensorboard_job_handle = None
tensorboard_enabled = True

TENSORBOARD_AVAILABLE = False
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    pass


def cleanup_on_exit():
    global training_process, is_training
    
    kill_all_training_processes()
    training_process = None
    is_training = False
    
    stop_tensorboard()


def signal_handler(signum, frame):
    cleanup_on_exit()
    sys.exit(0)


atexit.register(cleanup_on_exit)

try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except:
    pass


class LogCapture:
    def __init__(self, queue, original_stream):
        self.queue = queue
        self.original_stream = original_stream
    
    def write(self, text):
        if text:
            self.queue.put(text)
        self.original_stream.write(text)
    
    def flush(self):
        self.original_stream.flush()


def find_free_port(start_port=6006, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return start_port


def check_port_open(port, host='localhost', timeout=1):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _setup_windows_job_object(process):
    global tensorboard_job_handle
    
    try:
        import ctypes
        from ctypes import wintypes
        
        kernel32 = ctypes.windll.kernel32
        
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]
        
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]
        
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]
        
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        
        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        
        handle = kernel32.OpenProcess(0x1F0FFF, False, process.pid)
        if handle:
            kernel32.AssignProcessToJobObject(job, handle)
            kernel32.CloseHandle(handle)
        
        tensorboard_job_handle = job
    except Exception as e:
        print(f"  (Windows Job Object 设置失败: {e})")


def _get_linux_preexec_fn():
    def setup_process():
        try:
            os.setpgrp()
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except:
            pass
    return setup_process


def _kill_process_tree(process):
    if process is None:
        return
    
    try:
        if os.name != 'nt':
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                process.wait(timeout=5)
            except:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except:
                    pass
        else:
            process.terminate()
            process.wait(timeout=5)
    except:
        try:
            process.kill()
        except:
            pass


def start_tensorboard(port=None, logdir=None, force_restart=False):
    global tensorboard_process, tensorboard_port
    
    if not tensorboard_enabled:
        print("[config] TensorBoard 已禁用 (--no-tensorboard)")
        return
    
    if not TENSORBOARD_AVAILABLE:
        print("[!] TensorBoard 未安装，跳过启动")
        print("  如需使用 TensorBoard，请运行: pip install tensorboard")
        return
    
    if force_restart and tensorboard_process is not None:
        stop_tensorboard()
    
    if tensorboard_process is not None:
        print("TensorBoard 已在运行中")
        return
    
    tensorboard_dir = logdir if logdir else os.path.join(SCRIPT_DIR, "output", "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    requested_port = port if port is not None else 6006
    
    if check_port_open(requested_port):
        print(f"[!] 端口 {requested_port} 已被占用（可能有其他 TensorBoard 正在运行）")
        print(f"  正在查找可用端口...")
        tensorboard_port = find_free_port(requested_port + 1)
        print(f"  将使用端口 {tensorboard_port}")
    else:
        tensorboard_port = requested_port
    
    try:
        tensorboard_cmd = [
            sys.executable,
            '-m', 'tensorboard.main',
            '--logdir', tensorboard_dir,
            '--port', str(tensorboard_port),
            '--bind_all'
        ]
        
        popen_kwargs = {}
        if os.name == 'nt':
            popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs['preexec_fn'] = _get_linux_preexec_fn()
        
        tensorboard_process = subprocess.Popen(tensorboard_cmd, **popen_kwargs)
        
        if os.name == 'nt':
            _setup_windows_job_object(tensorboard_process)
        
        time.sleep(1)
        if tensorboard_process.poll() is not None:
            returncode = tensorboard_process.returncode
            tensorboard_process = None
            
            if returncode == -1 or returncode == 4294967295:
                print(f"[!] TensorBoard 启动失败：端口 {tensorboard_port} 可能被占用")
                print(f"  建议：")
                print(f"  1. 关闭其他正在运行的 TensorBoard 实例")
                print(f"  2. 使用不同的端口，如: --tensorboard {tensorboard_port + 10}")
                print(f"  3. 检查端口是否被其他程序占用")
            else:
                print(f"[!] TensorBoard 启动失败，返回码: {returncode}")
            return
        
        print(f"[ok] TensorBoard 已启动在端口 {tensorboard_port}")
        print(f"  日志目录: {tensorboard_dir}")
        print(f"  访问地址: http://localhost:{tensorboard_port}")
        print(f"  (子进程将随主进程退出)")
    except Exception as e:
        print(f"[!] TensorBoard 启动失败: {str(e)}")
        tensorboard_process = None


def stop_tensorboard():
    global tensorboard_process, tensorboard_job_handle
    
    if tensorboard_process is not None:
        try:
            tensorboard_process.terminate()
            tensorboard_process.wait(timeout=5)
            print("[ok] TensorBoard 已停止")
        except:
            try:
                tensorboard_process.kill()
            except:
                pass
        tensorboard_process = None
    
    if tensorboard_job_handle is not None and os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(tensorboard_job_handle)
        except:
            pass
        tensorboard_job_handle = None


def can_use_folder_dialog():
    return TKINTER_AVAILABLE and is_local_mode


def select_folder(current_path=""):
    if not can_use_folder_dialog():
        return current_path
    
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        initial_dir = current_path if current_path and os.path.exists(current_path) else None
        
        folder_path = filedialog.askdirectory(
            title="选择文件夹",
            initialdir=initial_dir
        )
        root.destroy()
        
        return folder_path if folder_path else current_path
    except Exception as e:
        print(f"文件夹选择失败: {e}")
        return current_path


def select_file(current_path="", filetypes=None):
    if not can_use_folder_dialog():
        return current_path
    
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        initial_dir = None
        if current_path:
            if os.path.isfile(current_path):
                initial_dir = os.path.dirname(current_path)
            elif os.path.isdir(current_path):
                initial_dir = current_path
        
        if filetypes is None:
            filetypes = [
                ("模型文件", "*.safetensors *.pth *.pt *.bin"),
                ("所有文件", "*.*")
            ]
        
        file_path = filedialog.askopenfilename(
            title="选择文件",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        root.destroy()
        
        return file_path if file_path else current_path
    except Exception as e:
        print(f"文件选择失败: {e}")
        return current_path


def stop_training():
    global should_stop, is_training, training_process
    
    if not is_training:
        print("\n[!] 当前没有正在运行的训练任务")
        return
    
    should_stop = True
    print("\n[stop] 用户请求停止训练...")
    
    stop_signal_file = os.path.join(SCRIPT_DIR, ".stop_training")
    stop_complete_file = os.path.join(SCRIPT_DIR, ".stop_complete")
    
    try:
        if os.path.exists(stop_complete_file):
            os.remove(stop_complete_file)
    except:
        pass
    
    try:
        with open(stop_signal_file, 'w') as f:
            f.write(str(time.time()))
        print("[ok] 已发送停止信号，等待模型保存...")
    except Exception as e:
        print(f"[!] 创建停止信号文件失败: {e}")
        return
    
    def wait_and_kill():
        last_status_time = time.time()
        
        while True:
            if os.path.exists(stop_complete_file):
                print("\n[ok] 模型保存完成，正在终止进程...")
                break
            
            if training_process is None or training_process.poll() is not None:
                print("\n[ok] 训练进程已退出")
                time.sleep(1)
                break
            
            if time.time() - last_status_time > 30:
                print("[...] 仍在等待模型保存完成...")
                last_status_time = time.time()
            
            time.sleep(0.5)
        
        kill_all_training_processes()
        
        try:
            if os.path.exists(stop_signal_file):
                os.remove(stop_signal_file)
            if os.path.exists(stop_complete_file):
                os.remove(stop_complete_file)
        except:
            pass
        
        print("[ok] 停止完成")
    
    kill_thread = threading.Thread(target=wait_and_kill, daemon=True)
    kill_thread.start()

