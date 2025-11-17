import os
import sys
import argparse
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)


def get_network(params):
    # Model selection based on arguments
    if params.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=3)
    elif params.network == "mobilenetv1":
        model = MobileNetV1(embedding_dim=512)
    elif params.network == "mobilenetv2":
        model = MobileNetV2(embedding_dim=512)
    elif params.network == "mobilenetv3_small":
        model = mobilenet_v3_small(embedding_dim=512)
    elif params.network == "mobilenetv3_large":
        model = mobilenet_v3_large(embedding_dim=512)
    else:
        raise ValueError("Unsupported network!")

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument(
        '-w', '--weights',
        default='./weights/mobilenetv2_mcp.pth',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'sphere20', 'sphere36', 'sphere64',
            'mobilenetv1', 'mobilenetv2',
            'mobilenetv3_small', 'mobilenetv3_large'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size and input dimensions for ONNX export'
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    print("=" * 70)
    print("ONNX EXPORT - Face Recognition Model")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")

    # Initialize model
    print(f"✓ Network: {params.network}")
    model = get_network(params)
    model.to(device)

    # Validate weights file exists
    print(f"\n📂 Loading weights from: {params.weights}")
    if not os.path.exists(params.weights):
        raise FileNotFoundError(f"Weights file not found: {params.weights}")

    # Load weights
    checkpoint = torch.load(params.weights, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        # É um checkpoint completo (.ckpt)
        state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        
        # Print additional checkpoint info if available
        if 'args' in checkpoint:
            ckpt_args = checkpoint['args']
            database = getattr(ckpt_args, 'database', 'N/A')
            classifier = getattr(ckpt_args, 'classifier', 'N/A')
            print(f"  - Database: {database}")
            print(f"  - Classifier: {classifier}")
    else:
        # É apenas state_dict (.pth)
        state_dict = checkpoint
        print("✓ Loaded state_dict (weights only)")

    model.load_state_dict(state_dict)
    print("✓ Model loaded successfully!")

    # Set model to evaluation mode
    model.eval()

    # Generate output filename in the same directory as weights
    output_dir = os.path.dirname(params.weights) or '.'
    fname = os.path.splitext(os.path.basename(params.weights))[0]
    onnx_model = os.path.join(output_dir, f'{fname}.onnx')
    
    print(f"\n💾 Output ONNX file: {onnx_model}")

    # Create dummy input tensor
    input_shape = (1, 3, 112, 112)
    x = torch.randn(input_shape).to(device)
    print(f"✓ Input shape: {input_shape}")

    # Prepare dynamic axes if --dynamic flag is enabled
    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print("✓ Dynamic batch size: ENABLED")
    else:
        print("✓ Dynamic batch size: DISABLED (fixed batch=1)")

    # Export model to ONNX
    print(f"\n🔄 Exporting to ONNX (opset version 16)...")
    
    torch.onnx.export(
        model,                      # PyTorch Model
        x,                          # Model input
        onnx_model,                 # Output file path
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=16,           # ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # Model's input names
        output_names=['output'],    # Model's output names
        dynamic_axes=dynamic_axes   # Use dynamic or static depending on flag
    )

    print("✅ ONNX export successful!")
    
    # Print file info
    file_size_mb = os.path.getsize(onnx_model) / (1024 * 1024)
    print(f"\n📊 Export Summary:")
    print(f"   - Output file: {onnx_model}")
    print(f"   - File size: {file_size_mb:.2f} MB")
    print(f"   - Network: {params.network}")
    print(f"   - Dynamic batch: {params.dynamic}")
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    try:
        args = parse_arguments()
        onnx_export(args)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)