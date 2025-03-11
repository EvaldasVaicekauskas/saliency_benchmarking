import pysaliency
import numpy as np
import os
import argparse

def load_fixations(fixation_path):
    """Load fixation data from a directory (example)."""
    return pysaliency.FixationTrains.from_matlab_files([fixation_path])

def evaluate_saliency(saliency_dir, fixations_dir):
    """Compute saliency evaluation metrics using pysaliency."""
    print("Loading fixation data...")
    fixations = load_fixations(os.path.join(fixations_dir, "fixations.mat"))

    print("Evaluating saliency maps...")
    for filename in os.listdir(saliency_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            saliency_map = pysaliency.SaliencyMapModel(
                lambda _: np.array(pysaliency.load_image(os.path.join(saliency_dir, filename)))
            )

            # Compute standard saliency metrics
            auc_judd = pysaliency.AUCJudd(saliency_map, fixations)
            nss = pysaliency.NSS(saliency_map, fixations)
            cc = pysaliency.CC(saliency_map, fixations)

            print(f"Results for {filename}:")
            print(f"AUC-Judd: {auc_judd}, NSS: {nss}, CC: {cc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saliency maps.")
    parser.add_argument("--saliency_dir", required=True, help="Path to saliency maps")
    parser.add_argument("--fixations_dir", required=True, help="Path to ground truth fixations")

    args = parser.parse_args()
    evaluate_saliency(args.saliency_dir, args.fixations_dir)