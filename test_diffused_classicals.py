import torch
import torch.nn as nn
import glob
import os
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from logistic_models.binary_classifier import LogisticClassifier
from logistic_models.Composer_Prediction.multi_composer import ComposerClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_genre_classification():
    """Test if diffused classical latents are classified as jazz"""
    print("=== Testing Genre Classification ===")
    
    # Load trained binary classifier
    binary_model = LogisticClassifier(128).to(device)
    binary_model.load_state_dict(torch.load("jazz_classical_classifier.pth", map_location=device))
    binary_model.eval()
    
    # Load all diffused classical latents
    diffused_files = glob.glob("diffused_classicals/*_diffused_latents.pt")
    all_predictions = []
    all_composers = []
    
    print(f"Found {len(diffused_files)} diffused composer files")
    
    for file_path in diffused_files:
        composer_name = os.path.basename(file_path).replace("_diffused_latents.pt", "")
        latents = torch.load(file_path, map_location=device)
        
        with torch.no_grad():
            predictions = binary_model.predict(latents)  # 1 = jazz, 0 = classical
            jazz_percentage = (predictions == 1).float().mean().item() * 100
            
        all_predictions.extend(predictions.cpu().numpy())
        all_composers.extend([composer_name] * len(latents))
        
        print(f"{composer_name}: {jazz_percentage:.1f}% classified as jazz ({len(latents)} samples)")
    
    overall_jazz_percentage = np.mean(all_predictions) * 100
    print(f"\nOverall: {overall_jazz_percentage:.1f}% of diffused classicals classified as jazz")
    
    return all_predictions, all_composers

def test_composer_classification():
    """Test if diffused classical latents retain composer identity"""
    print("\n=== Testing Composer Classification ===")
    
    # Load trained composer classifier
    composer_checkpoint = torch.load("composer_classifier_stratified.pth", map_location=device)
    composer_map = composer_checkpoint['composer_map']
    num_classes = composer_checkpoint['num_classes']
    
    composer_model = ComposerClassifier(128, num_classes).to(device)
    composer_model.load_state_dict(composer_checkpoint['model_state_dict'])
    composer_model.eval()
    
    # Reverse mapping for predictions
    idx_to_composer = {v: k for k, v in composer_map.items()}
    
    # Collect all predictions and true labels for confusion matrix
    all_true_labels = []
    all_predictions = []
    composer_results = {}
    
    # Test each diffused composer
    diffused_files = glob.glob("diffused_classicals/*_diffused_latents.pt")
    
    for file_path in diffused_files:
        true_composer = os.path.basename(file_path).replace("_diffused_latents.pt", "")
        
        if true_composer not in composer_map:
            print(f"Skipping {true_composer} - not in original training set")
            continue
            
        latents = torch.load(file_path, map_location=device)
        true_label = composer_map[true_composer]
        
        with torch.no_grad():
            outputs = composer_model(latents)
            _, predictions = torch.max(outputs, 1)
            
            # Calculate accuracy for this composer
            correct = (predictions == true_label).sum().item()
            accuracy = correct / len(latents) * 100
            
            # Store for confusion matrix
            all_true_labels.extend([true_label] * len(latents))
            all_predictions.extend(predictions.cpu().numpy())
            
            # Get prediction distribution
            pred_counts = {composer: 0 for composer in composer_map.keys()}
            for pred in predictions:
                pred_composer = idx_to_composer[pred.item()]
                pred_counts[pred_composer] += 1
            
            composer_results[true_composer] = {
                'accuracy': accuracy,
                'total_samples': len(latents),
                'predictions': pred_counts
            }
        
        print(f"{true_composer}: {accuracy:.1f}% correct identification ({correct}/{len(latents)})")
        
        # Show top 3 predicted composers
        top_predictions = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top predictions: {', '.join([f'{comp}({count})' for comp, count in top_predictions])}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, normalize='true')
    cm_percent = (cm * 100).astype(int)

    # print overall accuracy and F1 score
    overall_acc = accuracy_score(all_true_labels, all_predictions) * 100
    overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    print(f"\nOverall Composer Classification Accuracy: {overall_acc:.2f}%")
    print(f"Overall Composer Classification F1 Score: {overall_f1:.4f}")

    # naive baseline: predict uniformly at random
    random_preds = np.random.choice(list(composer_map.values()), size=len(all_true_labels))
    random_acc = accuracy_score(all_true_labels, random_preds) * 100
    print(f"Random Baseline Accuracy: {random_acc:.2f}%")
    
    return composer_results, cm_percent, list(composer_map.keys())

def plot_results(genre_predictions, composers, composer_results, cm_percent, composer_labels):
    """Plot the test results"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Genre classification results
    composer_names = list(set(composers))
    jazz_percentages = []
    
    for composer in composer_names:
        composer_preds = [genre_predictions[i] for i, c in enumerate(composers) if c == composer]
        jazz_pct = np.mean(composer_preds) * 100
        jazz_percentages.append(jazz_pct)
    
    bars1 = axes[0].bar(composer_names, jazz_percentages, color='lightblue', edgecolor='black')
    axes[0].set_title('Genre Classification: % Classified as Jazz')
    axes[0].set_ylabel('Percentage')
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars in black
    for bar, pct in zip(bars1, jazz_percentages):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    f'{pct:.1f}%', ha='center', va='center', fontweight='bold', color='black' if pct > 50 else 'red')
    
    # Plot 2: Composer classification confusion matrix
    if composer_results and len(cm_percent) > 0:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=composer_labels)
        disp.plot(ax=axes[1], xticks_rotation='vertical', cmap='Blues', values_format='d', colorbar=False)
        axes[1].set_title("Composer Identity Retention (%)")
    
    plt.tight_layout()
    plt.savefig('diffusion_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    if not os.path.exists("diffused_classicals"):
        print("Error: diffused_classicals folder not found. Run saving_diffused_classicals.py first.")
        return
    
    # Test genre classification
    genre_predictions, composers = test_genre_classification()
    
    # Test composer classification
    composer_results, cm_percent, composer_labels = test_composer_classification()
    
    # Plot results
    plot_results(genre_predictions, composers, composer_results, cm_percent, composer_labels)
    
    print(f"\nResults saved to 'diffusion_evaluation_results.png'")

if __name__ == "__main__":
    # set seed for reproducibility
    torch.manual_seed(42)
    main()



