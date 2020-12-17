import onnx
size = 256
if __name__ == "__main__":
    model = onnx.load_model("trained_model/bisenet-mobilenet-256.onnx")
    d = model.graph.input[0].type.tensor_type.shape.dim
    print(d)
    d[2].dim_value = size
    d[3].dim_value = size
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        d[2].dim_value = size
        d[3].dim_value = size
        print(d)
    onnx.save_model(model,"trained_model/bisenet-mobilenet-256-convert.onnx" )