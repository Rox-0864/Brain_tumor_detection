# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class TrainingPlotter:
    """
    Clase para graficar la historia de entrenamiento en ambas fases.
    """

    @staticmethod
    def plot(history_initial, history_fine=None, save_path=None):
        """
        Grafica la historia de entrenamiento para entrenamiento inicial y fine-tuning.

        :param history_initial: Objeto history del entrenamiento inicial.
        :param history_fine: Objeto history del fine-tuning (opcional).
        :param save_path: Ruta para guardar la figura (opcional).
        """
        if history_initial is None:
            print("No hay historial de entrenamiento disponible.")
            return

        # Función auxiliar para verificar existencia de keys en el history
        def has_keys(h, keys):
            return all(k in h.history for k in keys)

        keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        if not has_keys(history_initial, keys):
            print("El objeto history inicial no contiene las métricas requeridas.")
            return
        if history_fine is not None and not has_keys(history_fine, keys):
            print("El objeto history de fine-tuning no contiene las métricas requeridas.")
            return

        # Combinar resultados de entrenamiento y fine-tuning si hay
        if history_fine is not None:
            acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
            val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
            loss = history_initial.history['loss'] + history_fine.history['loss']
            val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
            transition_epoch = len(history_initial.epoch)
        else:
            acc = history_initial.history['accuracy']
            val_acc = history_initial.history['val_accuracy']
            loss = history_initial.history['loss']
            val_loss = history_initial.history['val_loss']
            transition_epoch = None

        epochs = range(1, len(acc) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Entrenamiento y Validación')

        # Accuracy plot
        axes[0].plot(epochs, acc, 'b-', label='Train Accuracy')
        axes[0].plot(epochs, val_acc, 'r-', label='Val Accuracy')
        if transition_epoch:
            axes[0].axvline(x=transition_epoch, color='g', linestyle='--', label='Fine-tuning start')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Precisión')
        axes[0].legend()
        axes[0].grid(True)

        # Loss plot
        axes[1].plot(epochs, loss, 'b-', label='Train Loss')
        axes[1].plot(epochs, val_loss, 'r-', label='Val Loss')
        if transition_epoch:
            axes[1].axvline(x=transition_epoch, color='g', linestyle='--', label='Fine-tuning start')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Pérdida')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path)
            print(f"Figura guardada en {save_path}")

        plt.show()