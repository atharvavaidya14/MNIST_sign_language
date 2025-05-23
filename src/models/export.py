import torch
from src.models.model_architecture import SimpleCNN


def export_model_torchscript(
    weights_path="trained_models/sign_cnn_best.pth",
    output_path="trained_models/sign_model_scripted.pt",
):
    """
    Export the trained model to TorchScript format."""
    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    example_input = torch.rand(1, 1, 28, 28)  # dummy input
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)
    print(f"TorchScript model saved to: {output_path}")


def export_model_onnx(
    weights_path="trained_models/sign_cnn_best.pth",
    output_path="trained_models/sign_model.onnx",
):
    """
    Export the trained model to ONNX format."""
    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    export_model_torchscript()
    export_model_onnx()
