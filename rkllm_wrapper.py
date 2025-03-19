import ctypes
from ctypes import c_int, c_char_p, c_void_p, c_float, Structure, POINTER, byref

# Load the RKLLM shared library
librkllm = ctypes.CDLL("librkllmrt.so")

# Define necessary structures and function prototypes
class RkllmModel(Structure):
    _fields_ = [("handle", c_void_p)]

class RkllmInput(Structure):
    _fields_ = [("text", c_char_p),
               ("length", c_int)]

class RkllmOutput(Structure):
    _fields_ = [("text", c_char_p),
               ("length", c_int)]

# Define function prototypes
librkllm.rkllm_init.argtypes = []
librkllm.rkllm_init.restype = c_int

librkllm.rkllm_load_model.argtypes = [c_char_p, POINTER(RkllmModel)]
librkllm.rkllm_load_model.restype = c_int

librkllm.rkllm_infer.argtypes = [POINTER(RkllmModel), POINTER(RkllmInput), POINTER(RkllmOutput)]
librkllm.rkllm_infer.restype = c_int

librkllm.rkllm_unload_model.argtypes = [POINTER(RkllmModel)]
librkllm.rkllm_unload_model.restype = c_int

librkllm.rkllm_deinit.argtypes = []
librkllm.rkllm_deinit.restype = c_int

# Wrapper class
class RkllmWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = RkllmModel()
        ret = librkllm.rkllm_init()
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKLLM runtime: {ret}")
        
        ret = librkllm.rkllm_load_model(model_path.encode(), byref(self.model))
        if ret != 0:
            raise RuntimeError(f"Failed to load model: {ret}")
    
    def generate_response(self, text):
        input_data = RkllmInput()
        input_data.text = text.encode()
        input_data.length = len(text)
        
        output_data = RkllmOutput()
        
        ret = librkllm.rkllm_infer(byref(self.model), byref(input_data), byref(output_data))
        if ret != 0:
            raise RuntimeError(f"Failed to run inference: {ret}")
            
        response = output_data.text.decode()
        return response
        
    def __del__(self):
        if hasattr(self, 'model'):
            librkllm.rkllm_unload_model(byref(self.model))
        librkllm.rkllm_deinit()
