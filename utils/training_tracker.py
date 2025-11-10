import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


class TrainingTracker:
    """
    Rastreia e visualiza métricas de treinamento completas
    
    Métricas rastreadas:
    - Loss de treino (por época)
    - Loss de validação (por época)
    - Accuracy de treino (por época)
    - Accuracy de validação (por época)
    - Métricas de validação externa (LFW/CelebA)
    - Learning rate (por época)
    - Face validation statistics (quando habilitado)
    """
    
    def __init__(self, save_dir, experiment_name=None):
        """
        Args:
            save_dir: Diretório para salvar logs e plots
            experiment_name: Nome do experimento (opcional)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Dicionários para armazenar métricas
        self.history = {
            # Métricas de treino
            'train_loss': [],
            'train_accuracy': [],
            
            # Métricas de validação interna
            'val_loss': [],
            'val_accuracy': [],
            
            # Métricas de validação externa (LFW/CelebA)
            'external_similarity': [],
            'external_accuracy': [],
            'external_f1': [],
            'external_precision': [],
            'external_recall': [],
            'external_auc': [],
            'external_best_threshold': [],
            
            # Confusion matrices por época
            'confusion_matrices': [],
            
            # Learning rate
            'learning_rate': [],
            
            # 🆕 Face validation statistics (quando RetinaFace está habilitado)
            'face_validation_enabled': False,
            'face_validation_policy': None,
            'face_validation_total_pairs': [],
            'face_validation_valid_pairs': [],
            'face_validation_excluded_pairs': [],
            'face_validation_exclusion_rate': [],
            
            # Época info
            'epochs': [],
            'epoch_times': []
        }
        
        self.start_time = None
        self.current_epoch = 0
        
        # Configurar estilo dos plots (estado-da-arte)
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configura estilo profissional para plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Configurações globais
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
    
    def log_epoch(self, epoch, train_loss, train_accuracy, val_accuracy=None, 
                  external_metrics=None, learning_rate=None, epoch_time=None):
        """
        Registra métricas de uma época
        
        Args:
            epoch: Número da época
            train_loss: Loss médio de treino
            train_accuracy: Accuracy médio de treino
            val_accuracy: Accuracy de validação interna (opcional)
            external_metrics: Dict com métricas LFW/CelebA (opcional)
            learning_rate: Learning rate atual (opcional)
            epoch_time: Tempo da época em segundos (opcional)
        """
        self.current_epoch = epoch
        self.history['epochs'].append(epoch)
        
        # Métricas de treino
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        
        # Validação interna
        if val_accuracy is not None:
            self.history['val_accuracy'].append(val_accuracy)
        
        # Learning rate
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
        
        # Tempo
        if epoch_time is not None:
            self.history['epoch_times'].append(epoch_time)
        
        # Métricas externas
        if external_metrics is not None:
            self.history['external_similarity'].append(
                external_metrics.get('mean_similarity', 0.0)
            )
            self.history['external_accuracy'].append(
                external_metrics.get('accuracy', 0.0)
            )
            self.history['external_f1'].append(
                external_metrics.get('f1', 0.0)
            )
            self.history['external_precision'].append(
                external_metrics.get('precision', 0.0)
            )
            self.history['external_recall'].append(
                external_metrics.get('recall', 0.0)
            )
            self.history['external_auc'].append(
                external_metrics.get('auc', 0.0)
            )
            self.history['external_best_threshold'].append(
                external_metrics.get('best_threshold', 0.35)
            )
            
            # Confusion matrix
            if 'confusion_matrix' in external_metrics:
                self.history['confusion_matrices'].append(
                    external_metrics['confusion_matrix'].tolist()
                )
            
            # 🆕 Face validation statistics
            if 'face_validation_stats' in external_metrics:
                face_stats = external_metrics['face_validation_stats']
                
                # Marcar que face validation está habilitado
                if not self.history['face_validation_enabled']:
                    self.history['face_validation_enabled'] = True
                
                # Log statistics
                self.history['face_validation_total_pairs'].append(
                    face_stats.get('total_pairs', 0)
                )
                self.history['face_validation_valid_pairs'].append(
                    face_stats.get('valid_pairs', 0)
                )
                self.history['face_validation_excluded_pairs'].append(
                    face_stats.get('excluded_pairs', 0)
                )
                
                # Calculate exclusion rate
                total = face_stats.get('total_pairs', 0)
                excluded = face_stats.get('excluded_pairs', 0)
                exclusion_rate = (excluded / total * 100) if total > 0 else 0.0
                self.history['face_validation_exclusion_rate'].append(exclusion_rate)
    
    def set_face_validation_policy(self, policy):
        """
        Define a política de face validation sendo usada
        
        Args:
            policy: 'exclude' ou 'include'
        """
        self.history['face_validation_policy'] = policy
    
    def plot_training_curves(self, save_path=None):
        """
        Gera plot profissional das curvas de treinamento
        
        Layout: 2x2 grid
        - Top-left: Train/Val Loss
        - Top-right: Train/Val Accuracy
        - Bottom-left: External Metrics (F1, Precision, Recall)
        - Bottom-right: AUC e Similarity
        """
        if len(self.history['epochs']) == 0:
            print("No data to plot yet.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = self.history['epochs']
        
        # Plot 1: Training & Validation Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 
                       marker='o', linewidth=2.5, label='Train Loss', color='#3498db')
        if len(self.history['val_loss']) > 0:
            axes[0, 0].plot(epochs, self.history['val_loss'], 
                           marker='s', linewidth=2.5, label='Val Loss', color='#e74c3c')
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold', pad=15)
        axes[0, 0].legend(loc='best', framealpha=0.9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight best loss
        if len(self.history['train_loss']) > 0:
            best_idx = np.argmin(self.history['train_loss'])
            axes[0, 0].axvline(x=epochs[best_idx], color='green', linestyle='--', 
                              alpha=0.5, label=f'Best (Epoch {epochs[best_idx]})')
            axes[0, 0].legend(loc='best', framealpha=0.9)
        
        # Plot 2: Accuracy Comparison
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 
                       marker='o', linewidth=2.5, label='Train Acc', color='#2ecc71')
        if len(self.history['val_accuracy']) > 0:
            axes[0, 1].plot(epochs, self.history['val_accuracy'], 
                           marker='s', linewidth=2.5, label='Val Acc', color='#f39c12')
        if len(self.history['external_accuracy']) > 0:
            axes[0, 1].plot(epochs, self.history['external_accuracy'], 
                           marker='^', linewidth=2.5, label='External Val Acc', color='#9b59b6')
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Accuracy Comparison', fontweight='bold', pad=15)
        axes[0, 1].legend(loc='best', framealpha=0.9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # Plot 3: Classification Metrics (F1, Precision, Recall)
        if len(self.history['external_f1']) > 0:
            axes[1, 0].plot(epochs, self.history['external_f1'], 
                           marker='o', linewidth=2.5, label='F1 Score', color='#e74c3c')
            axes[1, 0].plot(epochs, self.history['external_precision'], 
                           marker='s', linewidth=2.5, label='Precision', color='#3498db')
            axes[1, 0].plot(epochs, self.history['external_recall'], 
                           marker='^', linewidth=2.5, label='Recall', color='#2ecc71')
            axes[1, 0].set_xlabel('Epoch', fontweight='bold')
            axes[1, 0].set_ylabel('Score', fontweight='bold')
            axes[1, 0].set_title('Classification Metrics', fontweight='bold', pad=15)
            axes[1, 0].legend(loc='best', framealpha=0.9)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1.05])
        
        # Plot 4: AUC & Similarity (dual Y-axis)
        if len(self.history['external_auc']) > 0 or len(self.history['external_similarity']) > 0:
            ax1 = axes[1, 1]
            
            if len(self.history['external_auc']) > 0:
                color = '#9b59b6'
                ax1.plot(epochs, self.history['external_auc'], 
                        marker='o', linewidth=2.5, label='AUC Score', color=color)
                ax1.set_xlabel('Epoch', fontweight='bold')
                ax1.set_ylabel('AUC Score', color=color, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim([0.5, 1.05])
                ax1.grid(True, alpha=0.3)
            
            if len(self.history['external_similarity']) > 0:
                ax2 = ax1.twinx()
                color = '#e67e22'
                ax2.plot(epochs, self.history['external_similarity'], 
                        marker='s', linewidth=2.5, label='Mean Similarity', color=color)
                ax2.set_ylabel('Mean Similarity', color=color, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim([0, 1.0])
            
            axes[1, 1].set_title('AUC Score & Mean Similarity', fontweight='bold', pad=15)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            if len(self.history['external_similarity']) > 0:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
            else:
                ax1.legend(lines1, labels1, loc='best', framealpha=0.9)
        
        # Título geral
        fig.suptitle(f'Training Progress - {self.experiment_name}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path is None:
            save_path = self.save_dir / 'training_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Training curves saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_confusion_matrix_evolution(self, save_path=None):
        """
        Plota a evolução da confusion matrix em 3 pontos:
        início, meio e fim do treinamento
        """
        if len(self.history['confusion_matrices']) < 3:
            print("Need at least 3 epochs with confusion matrices to plot evolution.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Selecionar 3 pontos: início, meio, fim
        indices = [
            0,  # Início
            len(self.history['confusion_matrices']) // 2,  # Meio
            -1  # Fim
        ]
        
        titles = ['Early Training', 'Mid Training', 'Late Training']
        
        for idx, (cm_idx, title) in enumerate(zip(indices, titles)):
            cm = np.array(self.history['confusion_matrices'][cm_idx])
            epoch = self.history['epochs'][cm_idx]
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=True, ax=axes[idx],
                       xticklabels=['Different', 'Same'],
                       yticklabels=['Different', 'Same'])
            
            axes[idx].set_title(f'{title}\n(Epoch {epoch})', 
                              fontweight='bold', pad=15)
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
        
        fig.suptitle(f'Confusion Matrix Evolution - {self.experiment_name}', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'confusion_matrix_evolution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Confusion matrix evolution saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_learning_rate_schedule(self, save_path=None):
        """
        Plota o schedule do learning rate ao longo do treinamento
        """
        if len(self.history['learning_rate']) == 0:
            print("No learning rate data to plot.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = self.history['epochs']
        lrs = self.history['learning_rate']
        
        ax.plot(epochs, lrs, marker='o', linewidth=2.5, color='#e74c3c')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Anotar valores importantes
        for i in range(0, len(epochs), max(1, len(epochs)//5)):
            ax.annotate(f'{lrs[i]:.2e}', 
                       xy=(epochs[i], lrs[i]),
                       xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'learning_rate_schedule.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Learning rate schedule saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plota comparação lado-a-lado de todas as métricas principais
        Útil para análise final
        """
        if len(self.history['epochs']) == 0:
            print("No data to plot.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        epochs = self.history['epochs']
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'o-', linewidth=2)
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'o-', linewidth=2, label='Train')
        if len(self.history['external_accuracy']) > 0:
            axes[0, 1].plot(epochs, self.history['external_accuracy'], 's-', linewidth=2, label='External Val')
        axes[0, 1].set_title('Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # F1 Score
        if len(self.history['external_f1']) > 0:
            axes[0, 2].plot(epochs, self.history['external_f1'], 'o-', linewidth=2, color='orange')
            axes[0, 2].set_title('F1 Score', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('F1')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim([0, 1.05])
        
        # AUC
        if len(self.history['external_auc']) > 0:
            axes[1, 0].plot(epochs, self.history['external_auc'], 'o-', linewidth=2, color='purple')
            axes[1, 0].set_title('AUC Score', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0.5, 1.05])
        
        # Precision & Recall
        if len(self.history['external_precision']) > 0:
            axes[1, 1].plot(epochs, self.history['external_precision'], 'o-', linewidth=2, label='Precision')
            axes[1, 1].plot(epochs, self.history['external_recall'], 's-', linewidth=2, label='Recall')
            axes[1, 1].set_title('Precision & Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1.05])
        
        # Similarity
        if len(self.history['external_similarity']) > 0:
            axes[1, 2].plot(epochs, self.history['external_similarity'], 'o-', linewidth=2, color='red')
            axes[1, 2].set_title('Mean Similarity', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Similarity')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim([0, 1.0])
        
        fig.suptitle(f'All Metrics Overview - {self.experiment_name}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path is None:
            save_path = self.save_dir / 'all_metrics_overview.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ All metrics overview saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_face_validation_stats(self, save_path=None):
        """
        🆕 Plota estatísticas de face validation ao longo do treinamento
        Mostra a evolução de pares válidos vs excluídos
        """
        if not self.history['face_validation_enabled']:
            print("Face validation not enabled - no data to plot.")
            return
        
        if len(self.history['face_validation_total_pairs']) == 0:
            print("No face validation data to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = self.history['epochs'][:len(self.history['face_validation_total_pairs'])]
        
        # Plot 1: Valid vs Excluded Pairs
        axes[0].plot(epochs, self.history['face_validation_valid_pairs'], 
                    marker='o', linewidth=2.5, label='Valid Pairs', color='#2ecc71')
        axes[0].plot(epochs, self.history['face_validation_excluded_pairs'], 
                    marker='s', linewidth=2.5, label='Excluded Pairs', color='#e74c3c')
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Number of Pairs', fontweight='bold')
        axes[0].set_title('Face Validation: Valid vs Excluded Pairs', fontweight='bold', pad=15)
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Exclusion Rate
        axes[1].plot(epochs, self.history['face_validation_exclusion_rate'], 
                    marker='o', linewidth=2.5, color='#e67e22')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Exclusion Rate (%)', fontweight='bold')
        axes[1].set_title('Face Validation: Exclusion Rate Over Time', fontweight='bold', pad=15)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, max(self.history['face_validation_exclusion_rate']) * 1.2])
        
        # Add policy information
        policy_text = f"Policy: {self.history['face_validation_policy'].upper()}" if self.history['face_validation_policy'] else ""
        fig.text(0.5, 0.02, policy_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f'Face Validation Statistics - {self.experiment_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        if save_path is None:
            save_path = self.save_dir / 'face_validation_stats.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Face validation statistics saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def generate_final_report(self):
        """
        Gera relatório completo no final do treinamento
        Cria todos os plots e salva JSON com histórico
        """
        print("\n" + "="*70)
        print("GENERATING FINAL TRAINING REPORT")
        print("="*70 + "\n")
        
        report_dir = self.save_dir / 'final_report'
        report_dir.mkdir(exist_ok=True)
        
        # Gerar todos os plots
        plots = {}
        
        # 1. Training curves (principal)
        plots['training_curves'] = self.plot_training_curves(
            save_path=report_dir / 'training_curves.png'
        )
        
        # 2. Confusion matrix evolution
        plots['confusion_evolution'] = self.plot_confusion_matrix_evolution(
            save_path=report_dir / 'confusion_matrix_evolution.png'
        )
        
        # 3. Learning rate schedule
        plots['lr_schedule'] = self.plot_learning_rate_schedule(
            save_path=report_dir / 'learning_rate_schedule.png'
        )
        
        # 4. Metrics comparison
        plots['metrics_overview'] = self.plot_metrics_comparison(
            save_path=report_dir / 'all_metrics_overview.png'
        )
        
        # 5. 🆕 Face validation statistics (if enabled)
        if self.history['face_validation_enabled']:
            plots['face_validation'] = self.plot_face_validation_stats(
                save_path=report_dir / 'face_validation_stats.png'
            )
        
        # Salvar histórico completo em JSON
        history_json = report_dir / 'training_history.json'
        with open(history_json, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Training history saved to: {history_json}")
        
        # Gerar summary estatístico
        self._generate_summary_stats(report_dir)
        
        print("\n" + "="*70)
        print("FINAL REPORT GENERATED")
        print("="*70)
        print(f"📁 Report location: {report_dir}")
        print(f"📊 Training curves: {plots['training_curves'].name}")
        print(f"📋 Confusion evolution: {plots['confusion_evolution'].name}")
        print(f"📉 LR schedule: {plots['lr_schedule'].name}")
        print(f"📈 Metrics overview: {plots['metrics_overview'].name}")
        if self.history['face_validation_enabled']:
            print(f"🔍 Face validation stats: {plots['face_validation'].name}")
        print("="*70 + "\n")
        
        return report_dir
    
    def _generate_summary_stats(self, report_dir):
        """Gera arquivo de texto com estatísticas resumidas"""
        summary_file = report_dir / 'training_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"TRAINING SUMMARY - {self.experiment_name}\n")
            f.write("="*70 + "\n\n")
            
            # Info básica
            f.write(f"Total Epochs: {len(self.history['epochs'])}\n")
            if len(self.history['epoch_times']) > 0:
                total_time = sum(self.history['epoch_times'])
                avg_time = np.mean(self.history['epoch_times'])
                f.write(f"Total Training Time: {total_time/3600:.2f} hours\n")
                f.write(f"Average Epoch Time: {avg_time:.2f} seconds\n")
            f.write("\n")
            
            # 🆕 Face validation info
            if self.history['face_validation_enabled']:
                f.write("FACE VALIDATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Status: ENABLED\n")
                f.write(f"Policy: {self.history['face_validation_policy']}\n")
                if len(self.history['face_validation_total_pairs']) > 0:
                    avg_valid = np.mean(self.history['face_validation_valid_pairs'])
                    avg_excluded = np.mean(self.history['face_validation_excluded_pairs'])
                    avg_exclusion_rate = np.mean(self.history['face_validation_exclusion_rate'])
                    f.write(f"Average Valid Pairs: {avg_valid:.0f}\n")
                    f.write(f"Average Excluded Pairs: {avg_excluded:.0f}\n")
                    f.write(f"Average Exclusion Rate: {avg_exclusion_rate:.2f}%\n")
                f.write("\n")
            else:
                f.write("FACE VALIDATION: DISABLED\n\n")
            
            # Best metrics
            f.write("BEST METRICS:\n")
            f.write("-" * 40 + "\n")
            
            if len(self.history['train_loss']) > 0:
                best_loss_epoch = np.argmin(self.history['train_loss'])
                f.write(f"Best Train Loss: {self.history['train_loss'][best_loss_epoch]:.4f} "
                       f"(Epoch {self.history['epochs'][best_loss_epoch]})\n")
            
            if len(self.history['external_accuracy']) > 0:
                best_acc_epoch = np.argmax(self.history['external_accuracy'])
                f.write(f"Best External Accuracy: {self.history['external_accuracy'][best_acc_epoch]:.4f} "
                       f"(Epoch {self.history['epochs'][best_acc_epoch]})\n")
            
            if len(self.history['external_f1']) > 0:
                best_f1_epoch = np.argmax(self.history['external_f1'])
                f.write(f"Best F1 Score: {self.history['external_f1'][best_f1_epoch]:.4f} "
                       f"(Epoch {self.history['epochs'][best_f1_epoch]})\n")
            
            if len(self.history['external_auc']) > 0:
                best_auc_epoch = np.argmax(self.history['external_auc'])
                f.write(f"Best AUC Score: {self.history['external_auc'][best_auc_epoch]:.4f} "
                       f"(Epoch {self.history['epochs'][best_auc_epoch]})\n")
            
            f.write("\n")
            
            # Final metrics
            f.write("FINAL METRICS (Last Epoch):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"Train Accuracy: {self.history['train_accuracy'][-1]:.4f}\n")
            if len(self.history['external_accuracy']) > 0:
                f.write(f"External Accuracy: {self.history['external_accuracy'][-1]:.4f}\n")
                f.write(f"F1 Score: {self.history['external_f1'][-1]:.4f}\n")
                f.write(f"Precision: {self.history['external_precision'][-1]:.4f}\n")
                f.write(f"Recall: {self.history['external_recall'][-1]:.4f}\n")
                f.write(f"AUC Score: {self.history['external_auc'][-1]:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✅ Training summary saved to: {summary_file}")
    
    def save_checkpoint_with_history(self, checkpoint_dict, path):
        """
        Salva checkpoint incluindo histórico de treinamento
        
        Args:
            checkpoint_dict: Dict com checkpoint do PyTorch
            path: Caminho para salvar
        """
        import torch
        
        # Adicionar histórico ao checkpoint
        checkpoint_dict['training_history'] = self.history
        checkpoint_dict['tracker_experiment_name'] = self.experiment_name
        
        torch.save(checkpoint_dict, path)
        print(f"💾 Checkpoint with training history saved to: {path}")
    
    def load_history_from_checkpoint(self, checkpoint_path):
        """
        Carrega histórico de treinamento de um checkpoint
        
        Args:
            checkpoint_path: Caminho do checkpoint
            
        Returns:
            bool: True se histórico foi carregado com sucesso
        """
        import torch
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'training_history' in checkpoint:
                self.history = checkpoint['training_history']
                if 'tracker_experiment_name' in checkpoint:
                    self.experiment_name = checkpoint['tracker_experiment_name']
                print(f"✅ Training history loaded from checkpoint")
                print(f"   Epochs loaded: {len(self.history['epochs'])}")
                return True
            else:
                print("⚠️  No training history found in checkpoint")
                return False
        except Exception as e:
            print(f"❌ Failed to load training history: {e}")
            return False