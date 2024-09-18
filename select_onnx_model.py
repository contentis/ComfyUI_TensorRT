import os
import folder_paths

class ONNXModelSelector:
    @classmethod
    def INPUT_TYPES(s):
        onnx_path = os.path.join(folder_paths.models_dir,'onnx')
        if not os.path.exists(onnx_path):
            os.makedirs(onnx_path)
        onnx_models = [f for f in os.listdir(onnx_path) if f.endswith('.onnx')]
        return {
            "required": {
                "model_name": (onnx_models,),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "model_name")
    FUNCTION = "select_onnx_model"
    CATEGORY = "TensorRT"

    def select_onnx_model(self, model_name):
        onnx_path = os.path.join(folder_paths.models_dir,'onnx')
        model_path = os.path.join(onnx_path, model_name)
        return (model_path, model_name)

NODE_CLASS_MAPPINGS = {
    "ONNXModelSelector": ONNXModelSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ONNXModelSelector": "Select ONNX Model"
}