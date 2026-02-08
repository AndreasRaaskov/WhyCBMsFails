import torch
import numpy as np
from tqdm import tqdm


def Concept_cut_out(x, concept_coordinates, concept_idx, size=50):
    """
    Apply circular mask to cut out a concept from an image.

    Args:
        x: input image tensor of shape (3, H, W)
        concept_coordinates: list of tuples [(x1, y1), (x2, y2), ...] for each concept
        concept_idx: index of the concept to cut out
        size: diameter of the circular patch to cut out

    Returns:
        x_cut: image with circular mask applied
        cut_exists: boolean indicating if coordinates existed for this concept
    """
    x_cut = x.clone()
    cordinate_list = concept_coordinates[concept_idx]

    radius = size // 2

    if len(cordinate_list) == 0:
        return x_cut, False  # No coordinates for this concept, return original image

    def apply_circular_mask(img, x_center, y_center, radius):
        """Apply a circular black mask to the image"""
        H, W = img.shape[1], img.shape[2]

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # Calculate distance from center
        distances = torch.sqrt((x_grid - x_center)**2 + (y_grid - y_center)**2)

        # Create circular mask
        mask = distances <= radius

        # Apply mask to all channels
        img[:, mask] = 0.5

        return img

    if len(cordinate_list) == 1:
        x_center, y_center = cordinate_list[0]
        x_cut = apply_circular_mask(x_cut, x_center, y_center, radius)

    if len(cordinate_list) == 2:
        # Cut out both coordinates if double concepts (e.g., wings)
        x1, y1 = cordinate_list[0]
        x_cut = apply_circular_mask(x_cut, x1, y1, radius)

        x2, y2 = cordinate_list[1]
        x_cut = apply_circular_mask(x_cut, x2, y2, radius)

    return x_cut, True



def accuracy_analysis(dataset, XtoC_Model, CtoY_Model, concept_idx, mask_size=50, device='cpu', n_concepts=112):
    """
    Calculate exactly 4 accuracy metrics:
    1. Target concept accuracy (same class predictions only)
    2. Target concept accuracy (all images)
    3. Other concepts accuracy (same class predictions only)
    4. Class accuracy (all images)

    Returns original and masked accuracy for each metric.
    """
    XtoC_Model.eval()
    CtoY_Model.eval()
    XtoC_Model = XtoC_Model.to(device)
    CtoY_Model = CtoY_Model.to(device)

    # Image breakdown counters
    total_images = 0
    images_without_coords = 0
    same_class_pred_count = 0
    diff_class_pred_count = 0

    # Metrics for same class predictions
    same_pred_target_orig_correct = 0
    same_pred_target_mask_correct = 0
    same_pred_target_total = 0

    same_pred_other_orig_correct = 0
    same_pred_other_mask_correct = 0
    same_pred_other_total = 0

    # Metrics for all images
    all_target_orig_correct = 0
    all_target_mask_correct = 0
    all_target_total = 0

    all_class_orig_correct = 0
    all_class_mask_correct = 0
    all_class_total = 0

    # Confusion matrix for target concept (binary: 0 or 1)
    target_orig_tp = 0  # Predicted 1, True 1
    target_orig_fp = 0  # Predicted 1, True 0
    target_orig_tn = 0  # Predicted 0, True 0
    target_orig_fn = 0  # Predicted 0, True 1

    target_mask_tp = 0
    target_mask_fp = 0
    target_mask_tn = 0
    target_mask_fn = 0

    # Confusion matrices for SAME class predictions
    same_pred_target_orig_tp = 0
    same_pred_target_orig_fp = 0
    same_pred_target_orig_tn = 0
    same_pred_target_orig_fn = 0

    same_pred_target_mask_tp = 0
    same_pred_target_mask_fp = 0
    same_pred_target_mask_tn = 0
    same_pred_target_mask_fn = 0

    # Confusion matrices for DIFFERENT class predictions
    diff_pred_target_orig_tp = 0
    diff_pred_target_orig_fp = 0
    diff_pred_target_orig_tn = 0
    diff_pred_target_orig_fn = 0

    diff_pred_target_mask_tp = 0
    diff_pred_target_mask_fp = 0
    diff_pred_target_mask_tn = 0
    diff_pred_target_mask_fn = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Analyzing concept {concept_idx}"):
            x, c, y, coordinates = dataset[idx]
            total_images += 1

            # Try to mask the concept
            x_masked, cut_exists = Concept_cut_out(x, coordinates, concept_idx, size=mask_size)

            if not cut_exists:
                images_without_coords += 1
                continue

            # Prepare for model input
            x_batch = x.unsqueeze(0).to(device)
            x_masked_batch = x_masked.unsqueeze(0).to(device)

            # Get concept predictions
            concepts_original = torch.tensor(XtoC_Model(x_batch)).cpu()
            concepts_masked = torch.tensor(XtoC_Model(x_masked_batch)).cpu()

            # Get class predictions
            class_logits_original = CtoY_Model(concepts_original.to(device))
            class_logits_masked = CtoY_Model(concepts_masked.to(device))

            pred_class_original = torch.argmax(class_logits_original).item()
            pred_class_masked = torch.argmax(class_logits_masked).item()
            true_class = torch.argmax(y).cpu().item()

            # 4. Class accuracy on all images
            all_class_total += 1
            if pred_class_original == true_class:
                all_class_orig_correct += 1
            if pred_class_masked == true_class:
                all_class_mask_correct += 1

            # 2. Target concept accuracy on all images
            all_target_total += 1
            true_concept = c[concept_idx].item()
            pred_concept_orig = 1 if concepts_original[concept_idx].item() >= 0.5 else 0
            pred_concept_mask = 1 if concepts_masked[concept_idx].item() >= 0.5 else 0

            if pred_concept_orig == true_concept:
                all_target_orig_correct += 1
            if pred_concept_mask == true_concept:
                all_target_mask_correct += 1

            # Calculate confusion matrix for target concept
            true_concept_int = int(true_concept)
            # Original
            if pred_concept_orig == 1 and true_concept_int == 1:
                target_orig_tp += 1
            elif pred_concept_orig == 1 and true_concept_int == 0:
                target_orig_fp += 1
            elif pred_concept_orig == 0 and true_concept_int == 0:
                target_orig_tn += 1
            elif pred_concept_orig == 0 and true_concept_int == 1:
                target_orig_fn += 1

            # Masked
            if pred_concept_mask == 1 and true_concept_int == 1:
                target_mask_tp += 1
            elif pred_concept_mask == 1 and true_concept_int == 0:
                target_mask_fp += 1
            elif pred_concept_mask == 0 and true_concept_int == 0:
                target_mask_tn += 1
            elif pred_concept_mask == 0 and true_concept_int == 1:
                target_mask_fn += 1

            # Track same vs different class predictions
            if pred_class_original == pred_class_masked:
                same_class_pred_count += 1
            else:
                diff_class_pred_count += 1

            # Only analyze same class predictions
            if pred_class_original == pred_class_masked:
                # 1. Target concept accuracy (same class predictions)
                same_pred_target_total += 1
                if pred_concept_orig == true_concept:
                    same_pred_target_orig_correct += 1
                if pred_concept_mask == true_concept:
                    same_pred_target_mask_correct += 1

                # Confusion matrix for same class predictions
                # Original
                if pred_concept_orig == 1 and true_concept_int == 1:
                    same_pred_target_orig_tp += 1
                elif pred_concept_orig == 1 and true_concept_int == 0:
                    same_pred_target_orig_fp += 1
                elif pred_concept_orig == 0 and true_concept_int == 0:
                    same_pred_target_orig_tn += 1
                elif pred_concept_orig == 0 and true_concept_int == 1:
                    same_pred_target_orig_fn += 1

                # Masked
                if pred_concept_mask == 1 and true_concept_int == 1:
                    same_pred_target_mask_tp += 1
                elif pred_concept_mask == 1 and true_concept_int == 0:
                    same_pred_target_mask_fp += 1
                elif pred_concept_mask == 0 and true_concept_int == 0:
                    same_pred_target_mask_tn += 1
                elif pred_concept_mask == 0 and true_concept_int == 1:
                    same_pred_target_mask_fn += 1

                # 3. Other concepts accuracy (same class predictions)
                for other_idx in range(n_concepts):
                    if other_idx == concept_idx:
                        continue

                    true_other_concept = c[other_idx].item()
                    pred_other_orig = 1 if concepts_original[other_idx].item() >= 0.5 else 0
                    pred_other_mask = 1 if concepts_masked[other_idx].item() >= 0.5 else 0

                    same_pred_other_total += 1
                    if pred_other_orig == true_other_concept:
                        same_pred_other_orig_correct += 1
                    if pred_other_mask == true_other_concept:
                        same_pred_other_mask_correct += 1
            else:
                # Confusion matrix for different class predictions
                # Original
                if pred_concept_orig == 1 and true_concept_int == 1:
                    diff_pred_target_orig_tp += 1
                elif pred_concept_orig == 1 and true_concept_int == 0:
                    diff_pred_target_orig_fp += 1
                elif pred_concept_orig == 0 and true_concept_int == 0:
                    diff_pred_target_orig_tn += 1
                elif pred_concept_orig == 0 and true_concept_int == 1:
                    diff_pred_target_orig_fn += 1

                # Masked
                if pred_concept_mask == 1 and true_concept_int == 1:
                    diff_pred_target_mask_tp += 1
                elif pred_concept_mask == 1 and true_concept_int == 0:
                    diff_pred_target_mask_fp += 1
                elif pred_concept_mask == 0 and true_concept_int == 0:
                    diff_pred_target_mask_tn += 1
                elif pred_concept_mask == 0 and true_concept_int == 1:
                    diff_pred_target_mask_fn += 1

    # Calculate accuracies
    results = {
        # Image breakdown
        'total_images': total_images,
        'images_without_coords': images_without_coords,
        'same_class_pred_count': same_class_pred_count,
        'diff_class_pred_count': diff_class_pred_count,

        # 1. Target concept (same class predictions)
        '1_target_concept_same_pred_original': same_pred_target_orig_correct / same_pred_target_total if same_pred_target_total > 0 else 0,
        '1_target_concept_same_pred_masked': same_pred_target_mask_correct / same_pred_target_total if same_pred_target_total > 0 else 0,
        '1_target_concept_same_pred_count': same_pred_target_total,

        # 2. Target concept (all images) with confusion matrix
        '2_target_concept_all_images_original': all_target_orig_correct / all_target_total if all_target_total > 0 else 0,
        '2_target_concept_all_images_masked': all_target_mask_correct / all_target_total if all_target_total > 0 else 0,
        '2_target_concept_all_images_count': all_target_total,

        'target_orig_tp': target_orig_tp,
        'target_orig_fp': target_orig_fp,
        'target_orig_tn': target_orig_tn,
        'target_orig_fn': target_orig_fn,

        'target_mask_tp': target_mask_tp,
        'target_mask_fp': target_mask_fp,
        'target_mask_tn': target_mask_tn,
        'target_mask_fn': target_mask_fn,

        # Confusion matrices for SAME class predictions
        'same_pred_target_orig_tp': same_pred_target_orig_tp,
        'same_pred_target_orig_fp': same_pred_target_orig_fp,
        'same_pred_target_orig_tn': same_pred_target_orig_tn,
        'same_pred_target_orig_fn': same_pred_target_orig_fn,

        'same_pred_target_mask_tp': same_pred_target_mask_tp,
        'same_pred_target_mask_fp': same_pred_target_mask_fp,
        'same_pred_target_mask_tn': same_pred_target_mask_tn,
        'same_pred_target_mask_fn': same_pred_target_mask_fn,

        # Confusion matrices for DIFFERENT class predictions
        'diff_pred_target_orig_tp': diff_pred_target_orig_tp,
        'diff_pred_target_orig_fp': diff_pred_target_orig_fp,
        'diff_pred_target_orig_tn': diff_pred_target_orig_tn,
        'diff_pred_target_orig_fn': diff_pred_target_orig_fn,

        'diff_pred_target_mask_tp': diff_pred_target_mask_tp,
        'diff_pred_target_mask_fp': diff_pred_target_mask_fp,
        'diff_pred_target_mask_tn': diff_pred_target_mask_tn,
        'diff_pred_target_mask_fn': diff_pred_target_mask_fn,

        # 3. Other concepts (same class predictions)
        '3_other_concepts_same_pred_original': same_pred_other_orig_correct / same_pred_other_total if same_pred_other_total > 0 else 0,
        '3_other_concepts_same_pred_masked': same_pred_other_mask_correct / same_pred_other_total if same_pred_other_total > 0 else 0,
        '3_other_concepts_same_pred_count': same_pred_other_total,

        # 4. Class accuracy (all images)
        '4_class_all_images_original': all_class_orig_correct / all_class_total if all_class_total > 0 else 0,
        '4_class_all_images_masked': all_class_mask_correct / all_class_total if all_class_total > 0 else 0,
        '4_class_all_images_count': all_class_total,
        '4_class_orig_correct': all_class_orig_correct,
        '4_class_orig_incorrect': all_class_total - all_class_orig_correct,
        '4_class_mask_correct': all_class_mask_correct,
        '4_class_mask_incorrect': all_class_total - all_class_mask_correct,
    }

    return results


def simple_analysis(results, concept_idx, concept_name=None):
    """Print the 4 key accuracy metrics with detailed breakdown"""
    print(f"\n{'='*70}")
    print(f"Concept {concept_idx}" + (f" - {concept_name}" if concept_name else ""))
    print(f"{'='*70}")

    # Image breakdown
    print(f"\nIMAGE BREAKDOWN:")
    print(f"   Total images: {results['total_images']}")
    print(f"   Images without coordinates: {results['images_without_coords']}")
    print(f"   Images with SAME class prediction: {results['same_class_pred_count']}")
    print(f"   Images with DIFFERENT class prediction: {results['diff_class_pred_count']}")

    print(f"\n1. TARGET CONCEPT ACCURACY (Same Class Predictions)")
    print(f"   Images: {results['1_target_concept_same_pred_count']}")
    print(f"   Original: {results['1_target_concept_same_pred_original']:.4f}")
    print(f"   Masked:   {results['1_target_concept_same_pred_masked']:.4f}")

    print(f"\n2. TARGET CONCEPT ACCURACY (All Images)")
    print(f"   Images: {results['2_target_concept_all_images_count']}")
    print(f"   Original: {results['2_target_concept_all_images_original']:.4f}")
    print(f"   Masked:   {results['2_target_concept_all_images_masked']:.4f}")
    print(f"\n   Original Confusion Matrix (All Images):")
    print(f"      TP (Pred=1, True=1): {results['target_orig_tp']}")
    print(f"      FP (Pred=1, True=0): {results['target_orig_fp']}")
    print(f"      TN (Pred=0, True=0): {results['target_orig_tn']}")
    print(f"      FN (Pred=0, True=1): {results['target_orig_fn']}")
    print(f"\n   Masked Confusion Matrix (All Images):")
    print(f"      TP (Pred=1, True=1): {results['target_mask_tp']}")
    print(f"      FP (Pred=1, True=0): {results['target_mask_fp']}")
    print(f"      TN (Pred=0, True=0): {results['target_mask_tn']}")
    print(f"      FN (Pred=0, True=1): {results['target_mask_fn']}")

    print(f"\n   SAME Class Prediction - Original Confusion Matrix:")
    print(f"      TP (Pred=1, True=1): {results['same_pred_target_orig_tp']}")
    print(f"      FP (Pred=1, True=0): {results['same_pred_target_orig_fp']}")
    print(f"      TN (Pred=0, True=0): {results['same_pred_target_orig_tn']}")
    print(f"      FN (Pred=0, True=1): {results['same_pred_target_orig_fn']}")
    print(f"\n   SAME Class Prediction - Masked Confusion Matrix:")
    print(f"      TP (Pred=1, True=1): {results['same_pred_target_mask_tp']}")
    print(f"      FP (Pred=1, True=0): {results['same_pred_target_mask_fp']}")
    print(f"      TN (Pred=0, True=0): {results['same_pred_target_mask_tn']}")
    print(f"      FN (Pred=0, True=1): {results['same_pred_target_mask_fn']}")

    print(f"\n   DIFFERENT Class Prediction - Original Confusion Matrix:")
    print(f"      TP (Pred=1, True=1): {results['diff_pred_target_orig_tp']}")
    print(f"      FP (Pred=1, True=0): {results['diff_pred_target_orig_fp']}")
    print(f"      TN (Pred=0, True=0): {results['diff_pred_target_orig_tn']}")
    print(f"      FN (Pred=0, True=1): {results['diff_pred_target_orig_fn']}")
    print(f"\n   DIFFERENT Class Prediction - Masked Confusion Matrix:")
    print(f"      TP (Pred=1, True=1): {results['diff_pred_target_mask_tp']}")
    print(f"      FP (Pred=1, True=0): {results['diff_pred_target_mask_fp']}")
    print(f"      TN (Pred=0, True=0): {results['diff_pred_target_mask_tn']}")
    print(f"      FN (Pred=0, True=1): {results['diff_pred_target_mask_fn']}")

    print(f"\n3. OTHER CONCEPTS ACCURACY (Same Class Predictions)")
    print(f"   Predictions: {results['3_other_concepts_same_pred_count']}")
    print(f"   Original: {results['3_other_concepts_same_pred_original']:.4f}")
    print(f"   Masked:   {results['3_other_concepts_same_pred_masked']:.4f}")

    print(f"\n4. CLASS ACCURACY (All Images)")
    print(f"   Images: {results['4_class_all_images_count']}")
    print(f"   Original Accuracy: {results['4_class_all_images_original']:.4f}")
    print(f"   Masked Accuracy:   {results['4_class_all_images_masked']:.4f}")
    print(f"\n   Original: Correct={results['4_class_orig_correct']}, Incorrect={results['4_class_orig_incorrect']}")
    print(f"   Masked:   Correct={results['4_class_mask_correct']}, Incorrect={results['4_class_mask_incorrect']}")

    print(f"\n{'='*70}\n")
