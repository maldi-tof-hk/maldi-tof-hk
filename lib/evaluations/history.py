from lib.path import MetricsPath
import matplotlib.pyplot as plt

def evaluate_accuracy_history(history, path: MetricsPath):
    path = path.get_path("accuracy_history.png")
    print(f'Accuracy history saved to {path}')
    plt.clf()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylim(top=1, bottom=0)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower left')
    plt.savefig(path, dpi=300)

def evaluate_loss_history(history, path: MetricsPath):
    path = path.get_path("loss_history.png")
    print(f'Loss history saved to {path}')
    plt.clf()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylim(top=2)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig(path, dpi=300)

def evaluate_f1_history(history, path: MetricsPath):
    path = path.get_path("f1_history.png")
    print(f'F1 history saved to {path}')
    plt.clf()
    plt.plot(history['f_beta'])
    plt.plot(history['val_f_beta'])
    plt.title('Model F1')
    plt.ylim(top=1, bottom=0)
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower left')
    plt.savefig(path, dpi=300)

def evaluate_all_history(history, path: MetricsPath):
    evaluate_accuracy_history(history, path)
    evaluate_loss_history(history, path)
    evaluate_f1_history(history, path)