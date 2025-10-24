import os
import shutil
import argparse
import torch
from safetensors.torch import load_file, save_file
from glob import glob
from tqdm import tqdm

def merge_models(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory '{args.output_dir}' is ready.")

    print("\nScanning and merging .safetensors files...")

    b_files = glob(os.path.join(args.model_b, "*.safetensors"))
    i_files = glob(os.path.join(args.model_i, "*.safetensors"))
    i_tensors_all = {}

    for i_path in i_files:
        try:
            i_tensors = load_file(i_path, device="cpu")
        except Exception as e:
            print(f"Error: Failed to load model files: {e}")
            continue
        i_tensors_all.update(i_tensors)

    if not b_files:
        print(f"Warning: No .safetensors files found in directory '{args.model_b}'.")
        return

    for b_path in b_files:
        filename = os.path.basename(b_path)
        i_path = os.path.join(args.model_i, filename)
        output_path = os.path.join(args.output_dir, filename)

        if not os.path.exists(i_path):
            print(f"Warning: Corresponding file '{filename}' not found in '{args.model_i}', skipping.")
            continue

        print(f"\nMerging: \n  B: {b_path}\n  I: {i_path}")
        print(f"Lambda: {args.lambda_val}")

        try:
            b_tensors = load_file(b_path, device="cpu")
        except Exception as e:
            print(f"Error: Failed to load model files: {e}")
            continue

        merged_tensors = {}
        
        print("Computing merged weights...")
        for key in tqdm(b_tensors.keys(), desc=f"Merging {filename}"):
            if key in i_tensors_all:
                merged_tensors[key] = b_tensors[key] + args.lambda_val * (
                    i_tensors_all[key] - b_tensors[key]
                )
            else:
                print(f"  Warning: Weight '{key}' exists only in model B; copying directly.")
                merged_tensors[key] = b_tensors[key]

        print(f"Saving merged file to: {output_path}")
        try:
            save_file(merged_tensors, output_path)
        except Exception as e:
            print(f"Error: Failed to save merged file: {e}")
            continue

    print("\n.safetensors file merging completed.")


    print("\nChecking and copying other files...")
    for filename in os.listdir(args.model_b):
        if filename.endswith(".safetensors"):
            continue

        src_path = os.path.join(args.model_b, filename)
        dest_path = os.path.join(args.output_dir, filename)

        if not os.path.exists(dest_path):
            if os.path.isfile(src_path):
                print(f"Copying file: {filename}")
                shutil.copy2(src_path, dest_path)  # copy2 preserves metadata

    print("\ndone")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge weights from two models (B and I) and handle associated files.")

    parser.add_argument("--model_b", type=str, required=True, help="Path to the folder containing model B.")
    parser.add_argument("--model_i", type=str, required=True, help="Path to the folder containing model I.")
    parser.add_argument("--lambda_val", type=float, required=True, help="Value of the merging hyperparameter lambda.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the merged model and files.")

    args = parser.parse_args()

    merge_models(args)