import onnx
size = 384
if __name__ == "__main__":
    model = onnx.load_model("trained_model/bisenet-2020-11-18_23:03:39.onnx")
    d = model.graph.input[0].type.tensor_type.shape.dim
    print(d)
    d[2].dim_value = size
    d[3].dim_value = size
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        d[2].dim_value = size
        d[3].dim_value = size
        print(d)
    onnx.save_model(model,"trained_model/bisenet-dim-test.onnx" )