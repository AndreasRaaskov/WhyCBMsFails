"""
Experiment 2: Misclassification concept analysis

Hypothesis: When a model misclassifies an image (y_pred != y_true), the predicted
concepts will better match the concept pattern of the predicted (wrong) class
rather than the true class.

For each image:
- Run forward pass through XtoC and CtoY models
- If misclassified: compare predicted concepts with concepts[y_true] and concepts[y_pred]
- If correctly classified: use as baseline comparison

Metrics: Accuracy, Precision, Recall, F1 for both comparisons
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import data_loaders


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1 for binary predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def misclassification_analysis(dataset, XtoC_Model, CtoY_Model, device='cpu'):
    """
    Analyze concept predictions for correctly and incorrectly classified images.

    For misclassified images, compare predicted concepts against:
    1. True concepts (concepts associated with y_true)
    2. Predicted class concepts (concepts associated with y_pred)

    For correctly classified images, provide baseline metrics.
    """
    XtoC_Model.eval()
    CtoY_Model.eval()
    XtoC_Model = XtoC_Model.to(device)
    CtoY_Model = CtoY_Model.to(device)

    # Get the concept matrix from dataset (200 classes x 112 concepts)
    concept_matrix = dataset.concepts  # This is the majority-voted concept matrix

    n_concepts = len(dataset.consept_labels_names)
    concept_names = [str(n) for n in dataset.consept_labels_names]

    # Storage for predictions
    # Misclassified samples
    misclass_pred_concepts = []  # Model's predicted concepts
    misclass_true_concepts = []  # Concepts for y_true
    misclass_pred_class_concepts = []  # Concepts for y_pred
    misclass_info = []  # Store (y_true, y_pred) pairs for analysis

    # Correctly classified samples (baseline)
    correct_pred_concepts = []
    correct_true_concepts = []

    # Per-concept storage (append per-image concept vectors)
    misclass_pred_per_img = []
    misclass_true_per_img = []
    misclass_predclass_per_img = []
    correct_pred_per_img = []
    correct_true_per_img = []

    total_images = 0
    correct_count = 0
    misclass_count = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Analyzing misclassifications"):
            x, c, y, coordinates = dataset[idx]
            total_images += 1

            x_batch = x.unsqueeze(0).to(device)

            # Forward pass
            concepts_pred_logits = XtoC_Model(x_batch)
            concepts_pred_logits = torch.tensor(concepts_pred_logits).cpu()

            class_logits = CtoY_Model(concepts_pred_logits.to(device))
            pred_class = torch.argmax(class_logits).item()
            true_class = torch.argmax(y).cpu().item()

            # Threshold concepts at 0.5 for binary prediction
            concepts_pred_binary = (concepts_pred_logits.squeeze() >= 0.5).int().numpy()

            # Get concept patterns for true and predicted classes
            true_class_concepts = concept_matrix[true_class]
            pred_class_concepts = concept_matrix[pred_class]

            if pred_class == true_class:
                # Correctly classified
                correct_count += 1
                correct_pred_concepts.extend(concepts_pred_binary.tolist())
                correct_true_concepts.extend(true_class_concepts.tolist())
                correct_pred_per_img.append(concepts_pred_binary.tolist())
                correct_true_per_img.append(true_class_concepts.tolist())
            else:
                # Misclassified
                misclass_count += 1
                misclass_pred_concepts.extend(concepts_pred_binary.tolist())
                misclass_true_concepts.extend(true_class_concepts.tolist())
                misclass_pred_class_concepts.extend(pred_class_concepts.tolist())
                misclass_pred_per_img.append(concepts_pred_binary.tolist())
                misclass_true_per_img.append(true_class_concepts.tolist())
                misclass_predclass_per_img.append(pred_class_concepts.tolist())
                misclass_info.append({
                    'y_true': true_class,
                    'y_pred': pred_class,
                    'y_true_name': dataset.class_labels_names[true_class],
                    'y_pred_name': dataset.class_labels_names[pred_class],
                })

    # Compute metrics for misclassified samples
    misclass_vs_true = compute_metrics(misclass_true_concepts, misclass_pred_concepts)
    misclass_vs_pred_class = compute_metrics(misclass_pred_class_concepts, misclass_pred_concepts)

    # Compute baseline metrics for correctly classified samples
    correct_baseline = compute_metrics(correct_true_concepts, correct_pred_concepts)

    # Compute per-concept confusion matrices
    per_concept = {}
    if misclass_pred_per_img:
        misclass_pred_arr = np.array(misclass_pred_per_img)   # (n_misclass, n_concepts)
        misclass_true_arr = np.array(misclass_true_per_img)
        misclass_predclass_arr = np.array(misclass_predclass_per_img)
        correct_pred_arr = np.array(correct_pred_per_img)
        correct_true_arr = np.array(correct_true_per_img)

        for ci in range(n_concepts):
            pred_col = misclass_pred_arr[:, ci]
            true_col = misclass_true_arr[:, ci]
            predclass_col = misclass_predclass_arr[:, ci]
            correct_p = correct_pred_arr[:, ci]
            correct_t = correct_true_arr[:, ci]

            def confusion(y_true, y_pred):
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            per_concept[concept_names[ci]] = {
                'vs_true': confusion(true_col, pred_col),
                'vs_predicted_class': confusion(predclass_col, pred_col),
                'baseline': confusion(correct_t, correct_p),
            }

    results = {
        'summary': {
            'total_images': total_images,
            'correctly_classified': correct_count,
            'misclassified': misclass_count,
            'accuracy_rate': correct_count / total_images if total_images > 0 else 0,
        },

        'misclassified_vs_true_concepts': {
            'description': 'Predicted concepts compared to concepts of TRUE class (y_true)',
            'n_concept_comparisons': len(misclass_true_concepts),
            **misclass_vs_true,
        },

        'misclassified_vs_predicted_class_concepts': {
            'description': 'Predicted concepts compared to concepts of PREDICTED class (y_pred)',
            'n_concept_comparisons': len(misclass_pred_class_concepts),
            **misclass_vs_pred_class,
        },

        'correctly_classified_baseline': {
            'description': 'Baseline: Predicted concepts compared to true concepts for correctly classified images',
            'n_concept_comparisons': len(correct_true_concepts),
            **correct_baseline,
        },

        'misclassification_examples': misclass_info[:20],  # First 20 examples

        'per_concept': per_concept,
    }

    return results


def main():
    # Setup dataset (same as experiment_1)
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

    print("=" * 70)
    print("Experiment 2: Misclassification Concept Analysis")
    print("=" * 70)
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of concepts: {len(dataset.consept_labels_names)}")
    print(f"Number of classes: {dataset.n_classes}")
    print()

    # Run analysis
    results = misclassification_analysis(
        dataset=dataset,
        XtoC_Model=XtoC_Model,
        CtoY_Model=CtoY_Model,
        device='cpu',
    )

    # Create results directory if needed
    os.makedirs('results_experiment2', exist_ok=True)

    # Save results as JSON
    output_path = os.path.join('results_experiment2', 'misclassification_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = results['summary']
    print(f"\nImage Classification:")
    print(f"  Total images: {summary['total_images']}")
    print(f"  Correctly classified: {summary['correctly_classified']} ({summary['accuracy_rate']*100:.2f}%)")
    print(f"  Misclassified: {summary['misclassified']} ({(1-summary['accuracy_rate'])*100:.2f}%)")

    print(f"\n--- MISCLASSIFIED IMAGES ({summary['misclassified']} samples) ---")

    vs_true = results['misclassified_vs_true_concepts']
    print(f"\nPredicted concepts vs TRUE class concepts:")
    print(f"  Accuracy:  {vs_true['accuracy']*100:.2f}%")
    print(f"  Precision: {vs_true['precision']*100:.2f}%")
    print(f"  Recall:    {vs_true['recall']*100:.2f}%")
    print(f"  F1 Score:  {vs_true['f1']*100:.2f}%")

    vs_pred = results['misclassified_vs_predicted_class_concepts']
    print(f"\nPredicted concepts vs PREDICTED (wrong) class concepts:")
    print(f"  Accuracy:  {vs_pred['accuracy']*100:.2f}%")
    print(f"  Precision: {vs_pred['precision']*100:.2f}%")
    print(f"  Recall:    {vs_pred['recall']*100:.2f}%")
    print(f"  F1 Score:  {vs_pred['f1']*100:.2f}%")

    print(f"\n--- CORRECTLY CLASSIFIED BASELINE ({summary['correctly_classified']} samples) ---")
    baseline = results['correctly_classified_baseline']
    print(f"\nPredicted concepts vs TRUE class concepts:")
    print(f"  Accuracy:  {baseline['accuracy']*100:.2f}%")
    print(f"  Precision: {baseline['precision']*100:.2f}%")
    print(f"  Recall:    {baseline['recall']*100:.2f}%")
    print(f"  F1 Score:  {baseline['f1']*100:.2f}%")

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY FINDING:")
    print("=" * 70)
    acc_diff = vs_pred['accuracy'] - vs_true['accuracy']
    if acc_diff > 0:
        print(f"When misclassified, predicted concepts match the WRONG class better")
        print(f"by {acc_diff*100:.2f}% accuracy points.")
        print("This supports Hypothesis 2: The model produces concepts for the")
        print("predicted (wrong) class rather than the true class.")
    else:
        print(f"When misclassified, predicted concepts still match the TRUE class better")
        print(f"by {-acc_diff*100:.2f}% accuracy points.")
        print("This does NOT support Hypothesis 2.")


if __name__ == '__main__':
    main()
