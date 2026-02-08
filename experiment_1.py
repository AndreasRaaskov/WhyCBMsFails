"""
Experiment 1: Meta-concept analysis

For each meta concept (identified by splitting concept names at ::):
- Apply a single mask per image (all sub-concepts share the same body part location)
- Calculate accuracy metrics summed over all sub-concepts (target) and all other concepts
- Save results as JSON files in results/
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import data_loaders
import utils


def get_meta_concepts(concept_names):
    """Extract meta concepts by splitting at :: and group their indices."""
    meta_concepts = {}
    for idx, name in enumerate(concept_names):
        meta = name.split("::")[0]
        if meta not in meta_concepts:
            meta_concepts[meta] = []
        meta_concepts[meta].append(idx)
    return meta_concepts


def meta_concept_analysis(dataset, XtoC_Model, CtoY_Model, meta_name, concept_indices,
                          mask_size=50, device='cpu', n_concepts=112):
    """
    Run analysis for one meta concept.

    One forward pass per image since all sub-concepts in a meta concept share the same mask.
    Metrics are accumulated across all sub-concepts for the target, and across all
    non-member concepts for "other".
    """
    XtoC_Model.eval()
    CtoY_Model.eval()
    XtoC_Model = XtoC_Model.to(device)
    CtoY_Model = CtoY_Model.to(device)

    mask_concept_idx = concept_indices[0]  # All share same mask
    other_indices = [i for i in range(n_concepts) if i not in concept_indices]

    # Image breakdown
    total_images = 0
    images_without_coords = 0
    same_class_pred_count = 0
    diff_class_pred_count = 0

    # Target concept metrics (summed over all sub-concepts)
    # Same class predictions
    same_pred_target_orig_correct = 0
    same_pred_target_mask_correct = 0
    same_pred_target_total = 0

    # All images
    all_target_orig_correct = 0
    all_target_mask_correct = 0
    all_target_total = 0

    # Confusion matrices for target concept (all images)
    target_orig_tp = 0
    target_orig_fp = 0
    target_orig_tn = 0
    target_orig_fn = 0
    target_mask_tp = 0
    target_mask_fp = 0
    target_mask_tn = 0
    target_mask_fn = 0

    # Confusion matrices for target concept (same class pred)
    same_pred_target_orig_tp = 0
    same_pred_target_orig_fp = 0
    same_pred_target_orig_tn = 0
    same_pred_target_orig_fn = 0
    same_pred_target_mask_tp = 0
    same_pred_target_mask_fp = 0
    same_pred_target_mask_tn = 0
    same_pred_target_mask_fn = 0

    # Confusion matrices for target concept (diff class pred)
    diff_pred_target_orig_tp = 0
    diff_pred_target_orig_fp = 0
    diff_pred_target_orig_tn = 0
    diff_pred_target_orig_fn = 0
    diff_pred_target_mask_tp = 0
    diff_pred_target_mask_fp = 0
    diff_pred_target_mask_tn = 0
    diff_pred_target_mask_fn = 0

    # Other concept metrics (same class predictions)
    same_pred_other_orig_correct = 0
    same_pred_other_mask_correct = 0
    same_pred_other_total = 0

    # Other concept metrics (all images)
    all_other_orig_correct = 0
    all_other_mask_correct = 0
    all_other_total = 0

    # Class accuracy (all images)
    all_class_orig_correct = 0
    all_class_mask_correct = 0
    all_class_total = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Analyzing {meta_name}"):
            x, c, y, coordinates = dataset[idx]
            total_images += 1

            x_masked, cut_exists = utils.Concept_cut_out(x, coordinates, mask_concept_idx, size=mask_size)
            if not cut_exists:
                images_without_coords += 1
                continue

            x_batch = x.unsqueeze(0).to(device)
            x_masked_batch = x_masked.unsqueeze(0).to(device)

            # Single forward pass for original and masked
            concepts_original = torch.tensor(XtoC_Model(x_batch)).cpu()
            concepts_masked = torch.tensor(XtoC_Model(x_masked_batch)).cpu()

            class_logits_original = CtoY_Model(concepts_original.to(device))
            class_logits_masked = CtoY_Model(concepts_masked.to(device))

            pred_class_original = torch.argmax(class_logits_original).item()
            pred_class_masked = torch.argmax(class_logits_masked).item()
            true_class = torch.argmax(y).cpu().item()

            # Class accuracy (all images)
            all_class_total += 1
            if pred_class_original == true_class:
                all_class_orig_correct += 1
            if pred_class_masked == true_class:
                all_class_mask_correct += 1

            same_class = (pred_class_original == pred_class_masked)
            if same_class:
                same_class_pred_count += 1
            else:
                diff_class_pred_count += 1

            # Process all sub-concepts in the meta concept
            for cidx in concept_indices:
                true_concept = int(c[cidx].item())
                pred_orig = 1 if concepts_original[cidx].item() >= 0.5 else 0
                pred_mask = 1 if concepts_masked[cidx].item() >= 0.5 else 0

                # All images target accuracy
                all_target_total += 1
                if pred_orig == true_concept:
                    all_target_orig_correct += 1
                if pred_mask == true_concept:
                    all_target_mask_correct += 1

                # Confusion matrix (all images) - original
                if pred_orig == 1 and true_concept == 1:
                    target_orig_tp += 1
                elif pred_orig == 1 and true_concept == 0:
                    target_orig_fp += 1
                elif pred_orig == 0 and true_concept == 0:
                    target_orig_tn += 1
                elif pred_orig == 0 and true_concept == 1:
                    target_orig_fn += 1

                # Confusion matrix (all images) - masked
                if pred_mask == 1 and true_concept == 1:
                    target_mask_tp += 1
                elif pred_mask == 1 and true_concept == 0:
                    target_mask_fp += 1
                elif pred_mask == 0 and true_concept == 0:
                    target_mask_tn += 1
                elif pred_mask == 0 and true_concept == 1:
                    target_mask_fn += 1

                if same_class:
                    # Same class pred target accuracy
                    same_pred_target_total += 1
                    if pred_orig == true_concept:
                        same_pred_target_orig_correct += 1
                    if pred_mask == true_concept:
                        same_pred_target_mask_correct += 1

                    # Confusion matrix (same class) - original
                    if pred_orig == 1 and true_concept == 1:
                        same_pred_target_orig_tp += 1
                    elif pred_orig == 1 and true_concept == 0:
                        same_pred_target_orig_fp += 1
                    elif pred_orig == 0 and true_concept == 0:
                        same_pred_target_orig_tn += 1
                    elif pred_orig == 0 and true_concept == 1:
                        same_pred_target_orig_fn += 1

                    # Confusion matrix (same class) - masked
                    if pred_mask == 1 and true_concept == 1:
                        same_pred_target_mask_tp += 1
                    elif pred_mask == 1 and true_concept == 0:
                        same_pred_target_mask_fp += 1
                    elif pred_mask == 0 and true_concept == 0:
                        same_pred_target_mask_tn += 1
                    elif pred_mask == 0 and true_concept == 1:
                        same_pred_target_mask_fn += 1
                else:
                    # Confusion matrix (diff class) - original
                    if pred_orig == 1 and true_concept == 1:
                        diff_pred_target_orig_tp += 1
                    elif pred_orig == 1 and true_concept == 0:
                        diff_pred_target_orig_fp += 1
                    elif pred_orig == 0 and true_concept == 0:
                        diff_pred_target_orig_tn += 1
                    elif pred_orig == 0 and true_concept == 1:
                        diff_pred_target_orig_fn += 1

                    # Confusion matrix (diff class) - masked
                    if pred_mask == 1 and true_concept == 1:
                        diff_pred_target_mask_tp += 1
                    elif pred_mask == 1 and true_concept == 0:
                        diff_pred_target_mask_fp += 1
                    elif pred_mask == 0 and true_concept == 0:
                        diff_pred_target_mask_tn += 1
                    elif pred_mask == 0 and true_concept == 1:
                        diff_pred_target_mask_fn += 1

            # Process other concepts
            for oidx in other_indices:
                true_other = int(c[oidx].item())
                pred_other_orig = 1 if concepts_original[oidx].item() >= 0.5 else 0
                pred_other_mask = 1 if concepts_masked[oidx].item() >= 0.5 else 0

                # All images
                all_other_total += 1
                if pred_other_orig == true_other:
                    all_other_orig_correct += 1
                if pred_other_mask == true_other:
                    all_other_mask_correct += 1

                # Same class predictions
                if same_class:
                    same_pred_other_total += 1
                    if pred_other_orig == true_other:
                        same_pred_other_orig_correct += 1
                    if pred_other_mask == true_other:
                        same_pred_other_mask_correct += 1

    results = {
        'meta_concept': meta_name,
        'sub_concepts': [dataset.consept_labels_names[i] for i in concept_indices],
        'n_sub_concepts': len(concept_indices),

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

        # 3b. Other concepts (all images)
        '3b_other_concepts_all_images_original': all_other_orig_correct / all_other_total if all_other_total > 0 else 0,
        '3b_other_concepts_all_images_masked': all_other_mask_correct / all_other_total if all_other_total > 0 else 0,
        '3b_other_concepts_all_images_count': all_other_total,

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


def main():
    # Setup dataset (same as main.ipynb)
    majority_config = {
        'CUB_dir': r'CUB_200_2011',
        'split_file': r'CUB_200_2011/train_test_split.txt',
        'use_majority_voting': True,
        'min_class_count': 10,
        'return_visibility': False,
    }

    transform = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
    ])

    dataset = data_loaders.CUB_dataset(config_dict=majority_config, mode='test', transform=transform)

    # Load models
    XtoC_Model = torch.load(
        r'Models/ConceptModel__Seed1/outputs/best_model_1.pth',
        map_location=torch.device('cpu'), weights_only=False,
    )
    XtoC_Model.eval()

    CtoY_Model = torch.load(
        r'Models/IndependentModel_WithVal___Seed1/outputs/best_model_1.pth',
        map_location=torch.device('cpu'), weights_only=False,
    )
    CtoY_Model.eval()
    CtoY_Model = CtoY_Model.float()

    n_concepts = len(dataset.consept_labels_names)

    # Extract meta concepts
    meta_concepts = get_meta_concepts(dataset.consept_labels_names)
    print(f"Found {len(meta_concepts)} meta concepts:")
    for name, indices in meta_concepts.items():
        print(f"  {name}: {len(indices)} sub-concepts (indices {indices})")

    # Run analysis for each meta concept
    for meta_name, concept_indices in meta_concepts.items():
        print(f"\n{'='*70}")
        print(f"Processing meta concept: {meta_name}")
        print(f"Sub-concepts: {[dataset.consept_labels_names[i] for i in concept_indices]}")
        print(f"{'='*70}")

        results = meta_concept_analysis(
            dataset=dataset,
            XtoC_Model=XtoC_Model,
            CtoY_Model=CtoY_Model,
            meta_name=meta_name,
            concept_indices=concept_indices,
            mask_size=50,
            device='cpu',
            n_concepts=n_concepts,
        )

        # Save results as JSON
        output_path = os.path.join('results', f'{meta_name}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")

        # Print summary
        print(f"\nSummary for {meta_name}:")
        print(f"  Images analyzed: {results['4_class_all_images_count']} (skipped {results['images_without_coords']} without coords)")
        print(f"  Same/Diff class predictions: {results['same_class_pred_count']}/{results['diff_class_pred_count']}")
        print(f"  Target concept acc (same class): orig={results['1_target_concept_same_pred_original']:.4f}, masked={results['1_target_concept_same_pred_masked']:.4f}")
        print(f"  Target concept acc (all images): orig={results['2_target_concept_all_images_original']:.4f}, masked={results['2_target_concept_all_images_masked']:.4f}")
        print(f"  Other concepts acc (same class): orig={results['3_other_concepts_same_pred_original']:.4f}, masked={results['3_other_concepts_same_pred_masked']:.4f}")
        print(f"  Other concepts acc (all images): orig={results['3b_other_concepts_all_images_original']:.4f}, masked={results['3b_other_concepts_all_images_masked']:.4f}")
        print(f"  Class acc (all images):          orig={results['4_class_all_images_original']:.4f}, masked={results['4_class_all_images_masked']:.4f}")


if __name__ == '__main__':
    main()
