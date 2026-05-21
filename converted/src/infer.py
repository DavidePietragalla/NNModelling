import os
import sys
import json
import argparse

import torch
import lightning as lit
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from net.base import Net


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--config-path", default="cfg", help="Config directory (default: cfg)")
    parser.add_argument("--config-name", default="base", help="Config name (default: base)")
    parser.add_argument("--weights", default="weights.pt", help="Model weights (default: weights.pt)")
    parser.add_argument("--output", default=None, help="Save predictions to JSON file")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    # Resolve paths relative to converted/ (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    config_path = os.path.join(project_root, args.config_path)
    weights_path = os.path.join(project_root, args.weights)

    if not os.path.exists(config_path):
        print(f"Config directory not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    # Load composed Hydra config
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=args.config_name)

    # Load model and set to eval
    print(f"Loading model from {weights_path} ...")
    model = torch.load(weights_path, map_location=args.device, weights_only=False)
    model.eval()

    # Load dataset
    print("Loading dataset ...")
    dataset = instantiate(cfg.dataset)
    _, _, test_loader = dataset.division()

    # Run test loop via Lightning Trainer for metrics
    print("Running inference ...")
    try:
        trainer = lit.Trainer(logger=False, enable_progress_bar=True)
        results = trainer.test(model, test_loader)
        print("\nResults:")
        for key, value in results[0].items():
            print(f"  {key}: {value:.6f}")
    except Exception as e:
        print(f"  Metrics unavailable (loss function mismatch): {e}")

    # Save predictions if requested
    if args.output:
        print("\nCollecting predictions ...")
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(args.device)
                y_hat = model(x)
                # Argmax for classification, raw output for regression
                if y_hat.dim() > 1 and y_hat.size(1) > 1:
                    preds = y_hat.argmax(dim=1)
                else:
                    preds = y_hat
                for i in range(len(x)):
                    predictions.append({
                        "input": x[i].cpu().tolist(),
                        "target": y[i].cpu().tolist() if torch.is_tensor(y) else y[i],
                        "prediction": preds[i].cpu().tolist() if torch.is_tensor(preds) else preds[i],
                    })

        output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
