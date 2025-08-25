from typing import List, Optional
from dataclasses import dataclass
from src.utils.path_utils import format_time


def print_epoch_results(epoch, epochs, train_metrics, val_metrics, ar_metrics, epoch_time, lr, gradient_stats=None):
    """Print epoch results in a formatted table"""
    print(f"\nEPOCH {epoch+1}/{epochs} RESULTS ({format_time(epoch_time)})")
    print("┌─────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│    Phase    │   Loss   │ Accuracy │ExactMatch│ Windows  │    LR    │") 
    print("├─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    
    # Handle both old and new metric formats
    train_acc = train_metrics.get('accuracy', 0.0)
    train_exact = train_metrics.get('exact_match', 0.0)
    train_windows = train_metrics.get('total_windows', 0)
    
    print(f"│ Train       │ {train_metrics['loss']:8.4f} │ {train_acc:8.1%} │ {train_exact:8.1%} │ {train_windows:8d} │ {lr:8.2e} │")
    
    if val_metrics:
        val_acc = val_metrics.get('accuracy', 0.0)
        val_exact = val_metrics.get('exact_match', 0.0)
        val_windows = val_metrics.get('total_windows', 0)
        print(f"│ Validation  │ {val_metrics['loss']:8.4f} │ {val_acc:8.1%} │ {val_exact:8.1%} │ {val_windows:8d} │    -     │")

    if ar_metrics:
        ar_acc = ar_metrics.get('ar_accuracy', 0.0)
        ar_exact = ar_metrics.get('ar_exact_match', 0.0)
        print(f"│ Autoreg.    │ {ar_metrics['ar_loss']:8.4f} │ {ar_acc:8.1%} │ {ar_exact:8.1%} │    -     │    -     │")
        
    
    print("└─────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    
    # Показываем градиенты памяти если есть
    if gradient_stats:
        # Используем новую структуру блоков
        memory_norm = gradient_stats.get('gradients/blocks/memory/norm', 0.0)
        total_model_norm = gradient_stats.get('gradients/blocks/total_model/norm', 0.0)
        gating_norm = gradient_stats.get('gradients/blocks/gating/norm', 0.0)
        
        print(f"Gradient Blocks: Total: {total_model_norm:7.4f} │ Memory: {memory_norm:7.4f} │ Gating: {gating_norm:7.4f}")


@dataclass
class SlidingWindowMetrics:
    """Training metrics for sliding window approach"""
    loss: float = 0.0
    window_losses: Optional[List[float]] = None  # Loss per window
    exact_match_accuracy: float = 0.0
    token_accuracy: float = 0.0
    memory_utilization: float = 0.0
    memory_consistency: float = 0.0  # How consistent memory is across windows
    
    def __post_init__(self):
        if self.window_losses is None:
            self.window_losses = []
    
    def __str__(self) -> str:
        return (f"Loss: {self.loss:.4f}, "
                f"Exact Match: {self.exact_match_accuracy:.3f}, "
                f"Token Acc: {self.token_accuracy:.3f}, "
                f"Memory: {self.memory_utilization:.2f}")