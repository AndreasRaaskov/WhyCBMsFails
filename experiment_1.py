"""
Experiment 1: Meta-concept analysis

For each meta concept (identified by splitting concept names at ::):
- Apply a single mask per image (all sub-concepts share the same body part location)
- Calculate confusion matrices for manipulated concepts and other concepts
- Results structured as: {manipulated/other_concepts: {unmasked/masked: {all/same_pred/diff_pred: matrix}}}
- Save results as JSON files in results_experiment1/
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
        meta = name.split("_")[1]
        if meta not in meta_concepts:
            meta_concepts[meta] = []
        meta_concepts[meta].append(idx)
    return meta_concepts


def _empty_matrix():
    return {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}


def _update_matrix(matrix, pred, true):
    if pred == 1 and true == 1:
        matrix['tp'] += 1
    elif pred == 1 and true == 0:
        matrix['fp'] += 1
    elif pred == 0 and true == 0:
        matrix['tn'] += 1
    elif pred == 0 and true == 1:
        matrix['fn'] += 1


def _init_concept_matrices():
    """Create the {unmasked/masked: {all/same_pred/diff_pred: matrix}} structure."""
    return {
        'unmasked': {
            'all': _empty_matrix(),
            'same_pred': _empty_matrix(),
            'diff_pred': _empty_matrix(),
        },
        'masked': {
            'all': _empty_matrix(),
            'same_pred': _empty_matrix(),
            'diff_pred': _empty_matrix(),
        },
    }


def meta_concept_analysis(dataset, XtoC_Model, CtoY_Model, meta_name, concept_indices,
                          mask_size=100, device='cpu', n_concepts=112):
    """
    Run analysis for one meta concept.

    One forward pass per image since all sub-concepts in a meta concept share the same mask.
    Returns structured results with confusion matrices for both manipulated and other concepts.
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

    # Confusion matrices: {manipulated/other_concepts: {unmasked/masked: {all/same_pred/diff_pred: matrix}}}
    manipulated = _init_concept_matrices()
    other_concepts = _init_concept_matrices()

    # Class accuracy
    class_stats = {
        'unmasked': {'correct': 0, 'total': 0},
        'masked': {'correct': 0, 'total': 0},
    }

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

            # Class accuracy
            class_stats['unmasked']['total'] += 1
            class_stats['masked']['total'] += 1
            if pred_class_original == true_class:
                class_stats['unmasked']['correct'] += 1
            if pred_class_masked == true_class:
                class_stats['masked']['correct'] += 1

            same_class = (pred_class_original == pred_class_masked)
            if same_class:
                same_class_pred_count += 1
            else:
                diff_class_pred_count += 1

            subset = 'same_pred' if same_class else 'diff_pred'

            # Process manipulated concepts (target sub-concepts)
            for cidx in concept_indices:
                true_val = int(c[cidx].item())
                pred_orig = 1 if concepts_original[cidx].item() >= 0.5 else 0
                pred_mask = 1 if concepts_masked[cidx].item() >= 0.5 else 0

                _update_matrix(manipulated['unmasked']['all'], pred_orig, true_val)
                _update_matrix(manipulated['masked']['all'], pred_mask, true_val)
                _update_matrix(manipulated['unmasked'][subset], pred_orig, true_val)
                _update_matrix(manipulated['masked'][subset], pred_mask, true_val)

            # Process other concepts
            for oidx in other_indices:
                true_val = int(c[oidx].item())
                pred_orig = 1 if concepts_original[oidx].item() >= 0.5 else 0
                pred_mask = 1 if concepts_masked[oidx].item() >= 0.5 else 0

                _update_matrix(other_concepts['unmasked']['all'], pred_orig, true_val)
                _update_matrix(other_concepts['masked']['all'], pred_mask, true_val)
                _update_matrix(other_concepts['unmasked'][subset], pred_orig, true_val)
                _update_matrix(other_concepts['masked'][subset], pred_mask, true_val)

    results = {
        'meta_concept': meta_name,
        'sub_concepts': [dataset.consept_labels_names[i] for i in concept_indices],
        'n_sub_concepts': len(concept_indices),
        'total_images': total_images,
        'images_without_coords': images_without_coords,
        'same_class_pred_count': same_class_pred_count,
        'diff_class_pred_count': diff_class_pred_count,
        'manipulated': manipulated,
        'other_concepts': other_concepts,
        'class_accuracy': class_stats,
    }

    return results


def _acc(matrix):
    total = matrix['tp'] + matrix['fp'] + matrix['tn'] + matrix['fn']
    return (matrix['tp'] + matrix['tn']) / total if total > 0 else 0


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

    os.makedirs('results_experiment1', exist_ok=True)

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

        images_analyzed = results['total_images'] - results['images_without_coords']
        if images_analyzed == 0:
            print(f"  SKIPPED: No images had coordinates for {meta_name} (all {results['total_images']} skipped)")
            continue

        # Save results as JSON
        output_path = os.path.join('results_experiment1', f'{meta_name}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")

        # Print summary
        m = results['manipulated']
        o = results['other_concepts']
        cs = results['class_accuracy']
        print(f"\nSummary for {meta_name}:")
        print(f"  Images analyzed: {images_analyzed} (skipped {results['images_without_coords']} without coords)")
        print(f"  Same/Diff class predictions: {results['same_class_pred_count']}/{results['diff_class_pred_count']}")
        print(f"  Manipulated acc (all):       unmasked={_acc(m['unmasked']['all']):.4f}, masked={_acc(m['masked']['all']):.4f}")
        print(f"  Manipulated acc (same_pred): unmasked={_acc(m['unmasked']['same_pred']):.4f}, masked={_acc(m['masked']['same_pred']):.4f}")
        print(f"  Other acc (all):             unmasked={_acc(o['unmasked']['all']):.4f}, masked={_acc(o['masked']['all']):.4f}")
        print(f"  Other acc (same_pred):       unmasked={_acc(o['unmasked']['same_pred']):.4f}, masked={_acc(o['masked']['same_pred']):.4f}")
        print(f"  Class acc:                   unmasked={cs['unmasked']['correct']/cs['unmasked']['total']:.4f}, masked={cs['masked']['correct']/cs['masked']['total']:.4f}")


if __name__ == '__main__':
    main()
