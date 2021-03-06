import onnx
import sys
size = 256
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("缺少模型文件名")
    else:
        model_name = sys.argv[1]
        print("model name: " + model_name)
        model = onnx.load_model(model_name)
        d = model.graph.input[0].type.tensor_type.shape.dim
        print(d)
        d[2].dim_value = size
        d[3].dim_value = size
        for output in model.graph.output:
            d = output.type.tensor_type.shape.dim
            d[2].dim_value = size
            d[3].dim_value = size
            print(d)
        onnx.save_model(model,"convert.onnx" )