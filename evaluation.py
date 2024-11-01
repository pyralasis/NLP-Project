from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
import json

def evaluate_bert_model(model, data_loader, tag_enum, device='cuda'):
    """
    Evaluates BERT model predictions using a data loader
    
    Args:
        model: BERT model (BertForTokenClassification)
        data_loader: DataLoader instance with your SentimentDataset
        tag_enum: The Tag enumeration class
        device: Device to run evaluation on
    """
    model.eval()
    all_true_tags = []
    all_pred_tags = []
    
    with torch.no_grad():
        for texts, labels in data_loader:
            # Move inputs to device
            input_ids = texts['input_ids'].to(device)
            attention_mask = texts['attention_mask'].to(device)
            labels = labels.to(device)
            
            # Get BERT outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get predictions from logits
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Move to CPU for evaluation
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            
            # Process each sequence in the batch
            for pred, true, mask in zip(predictions, labels, attention_mask):
                # Filter out padding tokens
                valid_indices = mask == 1
                valid_pred = pred[valid_indices]
                valid_true = true[valid_indices]
                
                # Skip special tokens (-100)
                valid_indices = valid_true != -100
                valid_pred = valid_pred[valid_indices]
                valid_true = valid_true[valid_indices]
                
                all_pred_tags.extend(valid_pred)
                all_true_tags.extend(valid_true)
    
    # Calculate metrics
    tag_names = [tag.name for tag in tag_enum]
    report = classification_report(
        all_true_tags,
        all_pred_tags,
        labels=list(range(len(tag_enum))),
        target_names=tag_names,
        zero_division=0,
        output_dict=True
    )
    
    # Format results
    metrics = {
        'overall': {
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score']
        },
        'per_tag': {
            tag: {
                'precision': report[tag]['precision'],
                'recall': report[tag]['recall'],
                'f1': report[tag]['f1-score']
            }
            for tag in tag_names if tag in report
        }
    }
    
    return metrics

def print_metrics(metrics):
    """Print formatted metrics"""
    print("\n=== BERT Model Performance Metrics ===\n")
    
    print("Overall Metrics:")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {metrics['overall']['f1']:.4f}")
    
    print("\nPer-tag Metrics:")
    for tag, tag_metrics in metrics['per_tag'].items():
        print(f"\n{tag}:")
        print(f"  Precision: {tag_metrics['precision']:.4f}")
        print(f"  Recall: {tag_metrics['recall']:.4f}")
        print(f"  F1 Score: {tag_metrics['f1']:.4f}")


"""
Example main for using evaluator.
def main():
    from your_dataset_file import Tag, SentimentDataset
    
    # Initialize BERT model
    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(Tag)  # Number of tags in your enumeration
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load test dataset
    test_dataset = SentimentDataset(
        url='path_to_test_data',
        transform=tokenizer_custom,  # Your tokenizer function
        target_transform=None
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluate_bert_model(model, test_loader, Tag, device)
    print_metrics(metrics)
    
    # Save results
    with open('bert_evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
"""
